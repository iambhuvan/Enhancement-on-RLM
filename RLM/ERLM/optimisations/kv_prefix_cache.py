"""
O4: KV Cache Prefix Sharing via vLLM
======================================
Provides a vLLM-backed LM client that enables prefix caching
(RadixAttention).  When the RLM makes multiple subcalls that share a
long document or system-prompt prefix, vLLM reuses the already-computed
KV state for that prefix — eliminating redundant prefill computation and
dramatically reducing TTFT on the second and subsequent calls.

Architecture
------------
* ``VLLMPrefixCachedClient`` implements ``BaseLM`` and communicates with
  a running vLLM server via its OpenAI-compatible REST endpoint.
* Prefix caching is enabled server-side via ``--enable-prefix-caching``
  (RadixAttention) when the server is launched.
* ``get_cache_metrics()`` scrapes the Prometheus ``/metrics`` endpoint
  exposed by vLLM to surface hit rate, GPU cache utilisation, and
  prompt throughput.
* ``get_vllm_server_command()`` generates the shell command to start the
  server with the correct flags so experiments are reproducible.

Usage
-----
    client = create_vllm_client("Qwen/Qwen3-8B", port=8001)
    response = client.completion("Summarise the following document: ...")
    print(client.get_cache_metrics())
"""

import os
import sys
import time
from typing import Any

# ---------------------------------------------------------------------------
# Add BASELINE to sys.path so that rlm.* imports resolve correctly.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "BASELINE"))

from rlm.clients.base_lm import BaseLM  # noqa: E402  (after sys.path manipulation)
from rlm.core.types import ModelUsageSummary, UsageSummary  # noqa: E402

# ---------------------------------------------------------------------------
# Optional dependency: openai Python client
# ---------------------------------------------------------------------------
try:
    import openai as _openai_module

    _openai_available = True
except ImportError:
    _openai_module = None  # type: ignore[assignment]
    _openai_available = False

# ---------------------------------------------------------------------------
# Optional dependency: httpx for async metrics scraping
# ---------------------------------------------------------------------------
try:
    import httpx as _httpx

    _httpx_available = True
except ImportError:
    _httpx = None  # type: ignore[assignment]
    _httpx_available = False


# ---------------------------------------------------------------------------
# Prometheus text-format parser (minimal, no external deps)
# ---------------------------------------------------------------------------

def _parse_prometheus_metric(text: str, metric_name: str) -> float | None:
    """
    Extract the scalar value of a single Prometheus metric from a
    plain-text exposition dump.

    Only handles gauge / counter lines of the form::

        metric_name{...} <value>
        metric_name <value>

    Parameters
    ----------
    text:
        Raw Prometheus text-format response body.
    metric_name:
        Exact metric name to search for (e.g. ``"vllm:gpu_prefix_cache_hit_rate"``).

    Returns
    -------
    float | None
        The parsed float value, or ``None`` if the metric was not found or
        could not be parsed.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        # Match both "metric{labels} value" and "metric value"
        if stripped.startswith(metric_name):
            # The token after the metric name (and optional labels block) is the value
            remainder = stripped[len(metric_name):]
            # Skip optional labels: "{...}"
            if remainder.startswith("{"):
                closing = remainder.find("}")
                if closing == -1:
                    continue
                remainder = remainder[closing + 1:]
            parts = remainder.strip().split()
            if parts:
                try:
                    return float(parts[0])
                except ValueError:
                    continue
    return None


# ---------------------------------------------------------------------------
# VLLMPrefixCachedClient
# ---------------------------------------------------------------------------


class VLLMPrefixCachedClient(BaseLM):
    """
    LM client that targets a running vLLM server with prefix caching
    enabled.  Communicates through vLLM's OpenAI-compatible REST API.

    Prefix caching (RadixAttention) must be enabled on the server side via
    ``--enable-prefix-caching``.  See ``get_vllm_server_command()`` for the
    full launch command.

    Parameters
    ----------
    model_name:
        HuggingFace model ID served by vLLM (e.g. ``"Qwen/Qwen3-8B"``).
    base_url:
        Base URL of the vLLM OpenAI-compatible endpoint, including the
        ``/v1`` suffix (e.g. ``"http://localhost:8001/v1"``).
    api_key:
        API key sent in the Authorization header.  vLLM accepts any
        non-empty string when running without authentication.
    enable_prefix_caching:
        Informational flag — prefix caching is enforced server-side.
        Stored so callers can confirm the expected server config.
    quantization:
        Quantization scheme used when the server was launched (e.g.
        ``"fp8"``, ``"int8"``).  Stored for metadata purposes only.
    max_tokens:
        Maximum number of tokens to generate per completion.
    temperature:
        Sampling temperature.  Defaults to 0.0 (greedy decoding).
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        base_url: str = "http://localhost:8001/v1",
        api_key: str = "EMPTY",
        enable_prefix_caching: bool = True,
        quantization: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        timeout: float = 300.0,
    ) -> None:
        super().__init__(model_name=model_name, timeout=timeout)

        if not _openai_available:
            raise ImportError(
                "The 'openai' package is required for VLLMPrefixCachedClient. "
                "Install it with: pip install openai"
            )

        self.base_url: str = base_url
        self.api_key: str = api_key
        self.enable_prefix_caching: bool = enable_prefix_caching
        self.quantization: str | None = quantization
        self.max_tokens: int = max_tokens
        self.temperature: float = temperature

        # Derive the metrics endpoint URL (strip trailing /v1 if present)
        _base = base_url.rstrip("/")
        if _base.endswith("/v1"):
            self._metrics_base_url: str = _base[:-3]
        else:
            self._metrics_base_url = _base

        # Initialise the synchronous OpenAI client
        self._sync_client: "_openai_module.OpenAI" = _openai_module.OpenAI(  # type: ignore[name-defined]
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        # Initialise the asynchronous OpenAI client
        self._async_client: "_openai_module.AsyncOpenAI" = _openai_module.AsyncOpenAI(  # type: ignore[name-defined]
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

        # Usage tracking
        self.total_calls: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_ttft_seconds: list[float] = []

        # Last-call tracking (for get_last_usage())
        self._last_input_tokens: int = 0
        self._last_output_tokens: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _messages_from_prompt(self, prompt: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Normalise a prompt (str or OpenAI-style message list) into the
        OpenAI messages format.

        Parameters
        ----------
        prompt:
            Either a plain string (converted to a single ``user`` message)
            or an already-formatted list of role/content dicts.

        Returns
        -------
        list[dict[str, Any]]
            OpenAI-compatible messages list.
        """
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        if isinstance(prompt, list):
            return prompt
        raise ValueError(f"Unsupported prompt type: {type(prompt)!r}")

    def _record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        ttft: float,
    ) -> None:
        """Update cumulative and last-call usage counters."""
        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_ttft_seconds.append(ttft)
        self._last_input_tokens = input_tokens
        self._last_output_tokens = output_tokens

    # ------------------------------------------------------------------
    # BaseLM interface
    # ------------------------------------------------------------------

    def completion(self, prompt: str | list[dict[str, Any]]) -> str:
        """
        Synchronous completion via the vLLM OpenAI-compatible endpoint.

        For non-streaming requests, TTFT is approximated as total request
        latency (conservative upper bound).

        Parameters
        ----------
        prompt:
            Plain string or OpenAI messages list.

        Returns
        -------
        str
            The generated text content.
        """
        messages = self._messages_from_prompt(prompt)
        t_start = time.perf_counter()
        response = self._sync_client.chat.completions.create(
            model=self.model_name,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=False,
        )
        ttft = time.perf_counter() - t_start  # non-streaming: approx whole latency

        input_tokens = (response.usage.prompt_tokens or 0) if response.usage else 0
        output_tokens = (response.usage.completion_tokens or 0) if response.usage else 0
        self._record_usage(input_tokens, output_tokens, ttft)

        content = response.choices[0].message.content or ""
        return content

    async def acompletion(self, prompt: str | list[dict[str, Any]]) -> str:
        """
        Asynchronous completion via the vLLM OpenAI-compatible endpoint.

        Uses streaming to capture true TTFT (time to first token): we
        measure from request send to the moment the first chunk arrives,
        then accumulate the remaining chunks.

        Parameters
        ----------
        prompt:
            Plain string or OpenAI messages list.

        Returns
        -------
        str
            The generated text content.
        """
        messages = self._messages_from_prompt(prompt)
        t_start = time.perf_counter()
        first_token_recorded = False
        ttft: float = 0.0
        chunks: list[str] = []

        input_tokens: int = 0
        output_tokens: int = 0

        async with await self._async_client.chat.completions.create(
            model=self.model_name,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True,
            stream_options={"include_usage": True},
        ) as stream:
            async for chunk in stream:
                if not first_token_recorded:
                    # Record wall time to first token
                    delta_content = (
                        chunk.choices[0].delta.content
                        if chunk.choices and chunk.choices[0].delta
                        else None
                    )
                    if delta_content:
                        ttft = time.perf_counter() - t_start
                        first_token_recorded = True

                if chunk.choices and chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)

                # vLLM emits usage in the final chunk when stream_options includes it
                if chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens or 0
                    output_tokens = chunk.usage.completion_tokens or 0

        # If no streaming tokens arrived (empty response), record full latency as TTFT
        if not first_token_recorded:
            ttft = time.perf_counter() - t_start

        self._record_usage(input_tokens, output_tokens, ttft)
        return "".join(chunks)

    def get_usage_summary(self) -> UsageSummary:
        """
        Return a :class:`UsageSummary` with aggregate token counts across
        all calls made by this client instance.
        """
        return UsageSummary(
            model_usage_summaries={
                self.model_name: ModelUsageSummary(
                    total_calls=self.total_calls,
                    total_input_tokens=self.total_input_tokens,
                    total_output_tokens=self.total_output_tokens,
                )
            }
        )

    def get_last_usage(self) -> ModelUsageSummary:
        """Return token counts for the most recent completion call."""
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=self._last_input_tokens,
            total_output_tokens=self._last_output_tokens,
        )

    # ------------------------------------------------------------------
    # Cache / performance metrics
    # ------------------------------------------------------------------

    def avg_ttft(self) -> float:
        """
        Average time-to-first-token across all calls made so far.

        Returns
        -------
        float
            Mean TTFT in seconds, or 0.0 if no calls have been made.
        """
        if not self.total_ttft_seconds:
            return 0.0
        return sum(self.total_ttft_seconds) / len(self.total_ttft_seconds)

    def get_cache_metrics(self) -> dict[str, float]:
        """
        Scrape the vLLM Prometheus ``/metrics`` endpoint and return
        prefix-cache and throughput metrics.

        Metrics returned
        ----------------
        vllm:gpu_prefix_cache_hit_rate
            Fraction of prefill tokens served from the KV cache (0–1).
        vllm:gpu_cache_usage_perc
            Current GPU KV cache occupancy (0–1).
        vllm:avg_prompt_throughput_toks_per_s
            Rolling average prompt token ingestion rate.

        Returns
        -------
        dict[str, float]
            Populated dict on success, empty dict if the endpoint is
            unreachable or the ``httpx`` package is not installed.
        """
        metrics_url = f"{self._metrics_base_url}/metrics"

        # Prefer httpx; fall back to urllib (stdlib) if httpx is absent
        try:
            text = self._fetch_metrics_text(metrics_url)
        except Exception:
            return {}

        if text is None:
            return {}

        target_metrics = [
            "vllm:gpu_prefix_cache_hit_rate",
            "vllm:gpu_cache_usage_perc",
            "vllm:avg_prompt_throughput_toks_per_s",
        ]

        result: dict[str, float] = {}
        for name in target_metrics:
            value = _parse_prometheus_metric(text, name)
            if value is not None:
                result[name] = value

        return result

    def _fetch_metrics_text(self, url: str) -> str | None:
        """
        Fetch raw Prometheus text from ``url``.

        Tries ``httpx`` first (faster, connection pooling), then falls
        back to ``urllib.request``.

        Parameters
        ----------
        url:
            Full URL of the Prometheus metrics endpoint.

        Returns
        -------
        str | None
            Response body as text, or ``None`` on failure.
        """
        if _httpx_available:
            try:
                with _httpx.Client(timeout=5.0) as client:  # type: ignore[union-attr]
                    resp = client.get(url)
                    resp.raise_for_status()
                    return resp.text
            except Exception:
                pass

        # stdlib fallback
        try:
            import urllib.request

            with urllib.request.urlopen(url, timeout=5) as resp:
                return resp.read().decode("utf-8")
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def create_vllm_client(
    model_name: str = "Qwen/Qwen3-8B",
    quantization: str | None = None,
    enable_prefix_caching: bool = True,
    port: int = 8001,
) -> VLLMPrefixCachedClient:
    """
    Convenience factory for creating a :class:`VLLMPrefixCachedClient`.

    Parameters
    ----------
    model_name:
        HuggingFace model ID to query (must match the model served by vLLM).
    quantization:
        Quantization scheme the server was launched with (metadata only).
    enable_prefix_caching:
        Whether prefix caching is expected on the server (metadata only).
    port:
        Port the vLLM server is listening on.

    Returns
    -------
    VLLMPrefixCachedClient
        Ready-to-use client instance.
    """
    base_url = f"http://localhost:{port}/v1"
    return VLLMPrefixCachedClient(
        model_name=model_name,
        base_url=base_url,
        api_key="EMPTY",
        enable_prefix_caching=enable_prefix_caching,
        quantization=quantization,
    )


def get_vllm_server_command(
    model_name: str = "Qwen/Qwen3-8B",
    port: int = 8001,
    quantization: str | None = None,
    enable_prefix_caching: bool = True,
) -> str:
    """
    Return the shell command to launch a vLLM server with the configuration
    expected by :class:`VLLMPrefixCachedClient`.

    Parameters
    ----------
    model_name:
        HuggingFace model ID to serve.
    port:
        Port to bind the HTTP server on.
    quantization:
        Quantization flag passed to ``--quantization`` (e.g. ``"fp8"``).
        Omitted from the command when ``None``.
    enable_prefix_caching:
        When ``True``, adds ``--enable-prefix-caching`` (RadixAttention).

    Returns
    -------
    str
        Shell command string that can be run directly or printed for the
        user to execute in a separate terminal / subprocess.

    Example
    -------
    >>> print(get_vllm_server_command("Qwen/Qwen3-8B", port=8001, quantization="fp8"))
    vllm serve Qwen/Qwen3-8B --port 8001 --enable-prefix-caching \
        --quantization fp8 --gpu-memory-utilization 0.85
    """
    parts = [
        "vllm",
        "serve",
        model_name,
        "--port",
        str(port),
    ]

    if enable_prefix_caching:
        parts.append("--enable-prefix-caching")

    if quantization is not None:
        parts += ["--quantization", quantization]

    parts += ["--gpu-memory-utilization", "0.85"]

    return " ".join(parts)
