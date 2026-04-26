"""
Vertex AI Client for Qwen3-Coder-480B-A35B
============================================
Wraps the BASELINE OpenAIClient with Vertex AI authentication.

Vertex AI exposes an OpenAI-compatible endpoint:
  https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/{REGION}/endpoints/openapi

Auth: Google access token, refreshed automatically via `google-auth`.

Usage:
    client = VertexAIClient(project_id="my-gcp-project")
    response = client.completion("Explain this code...")

Env vars (alternative to constructor args):
    VERTEX_PROJECT_ID   — GCP project ID
    VERTEX_REGION       — e.g. "us-central1" (default)
    GOOGLE_API_KEY      — if set, skips ADC auth (useful for testing)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

_log = logging.getLogger(__name__)

# Retry config for 429 / transient errors inside the client
_CLIENT_MAX_RETRIES = 6
_CLIENT_BASE_WAIT = 60  # seconds; doubles each retry (60, 120, 240, …)

# Add BASELINE to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "BASELINE"))

from rlm.clients.openai import OpenAIClient  # noqa: E402

# Qwen3-Coder model ID as registered on Vertex AI Model Garden
QWEN3_CODER_480B = "qwen/qwen3-coder-480b-a35b-instruct-maas"

# How often to refresh the access token (seconds); tokens expire after 3600s
_TOKEN_REFRESH_INTERVAL = 3000


def _get_vertex_access_token() -> str:
    """
    Get a short-lived Google OAuth2 access token using Application Default
    Credentials (ADC).  Requires either:
      - `gcloud auth application-default login` to have been run, OR
      - GOOGLE_APPLICATION_CREDENTIALS env var pointing to a service-account key.

    Returns
    -------
    str
        Bearer token string.

    Raises
    ------
    ImportError
        If `google-auth` is not installed.
    RuntimeError
        If credentials cannot be obtained.
    """
    try:
        import google.auth
        import google.auth.transport.requests
    except ImportError as exc:
        raise ImportError(
            "google-auth is required for VertexAIClient. "
            "Install with: pip install google-auth"
        ) from exc

    try:
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        request = google.auth.transport.requests.Request()
        credentials.refresh(request)
        return credentials.token
    except Exception as exc:
        raise RuntimeError(
            f"Failed to obtain Vertex AI credentials: {exc}\n"
            "Run: gcloud auth application-default login"
        ) from exc


class VertexAIClient(OpenAIClient):
    """
    LM client targeting Vertex AI's OpenAI-compatible endpoint.

    Inherits all token tracking and async support from OpenAIClient.
    Automatically refreshes the Google auth token before it expires.

    Parameters
    ----------
    project_id:
        GCP project ID (overrides VERTEX_PROJECT_ID env var).
    region:
        GCP region where the model is deployed (default: ``"us-central1"``).
    model_name:
        Vertex AI model garden ID (default: Qwen3-Coder-480B-A35B).
    max_tokens:
        Max output tokens per completion.
    temperature:
        Sampling temperature (0.0 = greedy).
    """

    def __init__(
        self,
        project_id: str | None = None,
        region: str = "us-south1",
        model_name: str = QWEN3_CODER_480B,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs,
    ) -> None:
        self.project_id = project_id or os.environ.get("VERTEX_PROJECT_ID")
        if not self.project_id:
            raise ValueError(
                "project_id is required. Pass it directly or set VERTEX_PROJECT_ID."
            )
        self.region = region or os.environ.get("VERTEX_REGION", "us-south1")
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Build Vertex AI OpenAI-compatible base URL
        base_url = (
            f"https://{self.region}-aiplatform.googleapis.com/v1/projects/"
            f"{self.project_id}/locations/{self.region}/endpoints/openapi"
        )

        # Get initial access token
        self._token: str = _get_vertex_access_token()
        self._token_fetched_at: float = time.time()

        super().__init__(
            api_key=self._token,
            model_name=model_name,
            base_url=base_url,
            **kwargs,
        )

    def _refresh_token_if_needed(self) -> None:
        """Refresh the access token if it is about to expire."""
        if time.time() - self._token_fetched_at > _TOKEN_REFRESH_INTERVAL:
            self._token = _get_vertex_access_token()
            self._token_fetched_at = time.time()
            # Patch the underlying openai client with the new token
            self.client.api_key = self._token
            self.async_client.api_key = self._token

    def _inject_no_think(self, prompt: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Inject /no_think into the system prompt to disable Qwen3 thinking mode globally.

        Thinking mode generates ~400K internal tokens per call, exhausting TPM quota rapidly.
        Placing /no_think in the system prompt is the most reliable approach: it applies to
        every call (main loop, sub-calls, _default_answer) without depending on which role
        the last message has.
        """
        if isinstance(prompt, str):
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": "/no_think"},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = list(prompt)  # shallow copy
            # Check if there's already a system message
            has_system = any(m.get("role") == "system" for m in messages)
            if has_system:
                # Prepend /no_think to the existing system message
                new_messages = []
                for m in messages:
                    if m.get("role") == "system":
                        m = dict(m)
                        if "/no_think" not in m.get("content", ""):
                            m["content"] = "/no_think\n" + m["content"]
                    new_messages.append(m)
                messages = new_messages
            else:
                # Insert a system message at the front
                messages = [{"role": "system", "content": "/no_think"}] + messages
        return messages

    def _is_rate_limit(self, exc: Exception) -> bool:
        msg = str(exc)
        return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "rate limit" in msg.lower()

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        self._refresh_token_if_needed()
        messages = self._inject_no_think(prompt)
        model = model or self.model_name
        wait = _CLIENT_BASE_WAIT
        last_exc: Exception | None = None
        for attempt in range(_CLIENT_MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    extra_body={"enable_thinking": False},
                )
                self._track_cost(response, model)
                return response.choices[0].message.content
            except Exception as exc:
                if self._is_rate_limit(exc):
                    last_exc = exc
                    _log.warning(
                        f"[VertexAI] Rate limit (attempt {attempt+1}/{_CLIENT_MAX_RETRIES}). "
                        f"Waiting {wait}s… ({exc})"
                    )
                    time.sleep(wait)
                    wait = min(wait * 2, 480)
                    self._refresh_token_if_needed()
                else:
                    raise
        raise last_exc  # type: ignore[misc]

    async def acompletion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        import asyncio
        self._refresh_token_if_needed()
        messages = self._inject_no_think(prompt)
        model = model or self.model_name
        wait = _CLIENT_BASE_WAIT
        last_exc: Exception | None = None
        for attempt in range(_CLIENT_MAX_RETRIES):
            try:
                response = await self.async_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    extra_body={"enable_thinking": False},
                )
                self._track_cost(response, model)
                return response.choices[0].message.content
            except Exception as exc:
                if self._is_rate_limit(exc):
                    last_exc = exc
                    _log.warning(
                        f"[VertexAI] Async rate limit (attempt {attempt+1}/{_CLIENT_MAX_RETRIES}). "
                        f"Waiting {wait}s… ({exc})"
                    )
                    await asyncio.sleep(wait)
                    wait = min(wait * 2, 480)
                    self._refresh_token_if_needed()
                else:
                    raise
        raise last_exc  # type: ignore[misc]
