"""
O5: FP8 / INT8 Quantization Utilities
========================================
Measurement utilities and configuration helpers for running quantized
models via vLLM.

Background
----------
* **FP8** (E4M3 / E5M2): Natively accelerated on NVIDIA H100 Tensor
  Cores.  Delivers ~2x memory savings over BF16/FP16 and ~2x throughput
  improvement with near-identical accuracy.  Enabled in vLLM with
  ``--quantization fp8``.
* **INT8** (LLM.int8 / SmoothQuant): Works on A100 and later.  ~2x
  memory savings, ~1.5x throughput gain.  Enabled with
  ``--quantization int8``.
* **INT4** (GPTQ / AWQ): 4x memory savings, highest throughput boost,
  but accuracy degrades on difficult reasoning tasks.

These classes do **not** perform quantization themselves — they
configure, benchmark, and recommend the correct vLLM flag so that the
vLLM server (or the ``VLLMPrefixCachedClient`` in kv_prefix_cache.py)
can launch with the optimal scheme for the available hardware.

All classes are instantiable without a GPU — pure Python, no CUDA
required at import time.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Optional dependency: torch (needed only for GPU introspection helpers)
# ---------------------------------------------------------------------------
try:
    import torch as _torch

    _torch_available = True
except ImportError:
    _torch = None  # type: ignore[assignment]
    _torch_available = False


# ---------------------------------------------------------------------------
# Constants: theoretical multipliers vs FP16 baseline
# ---------------------------------------------------------------------------

_MEMORY_REDUCTION: dict[str, float] = {
    "none": 1.0,   # FP16 baseline — no reduction
    "int8": 0.5,   # 8-bit vs 16-bit → half the memory
    "fp8":  0.5,   # 8-bit vs 16-bit → half the memory
    "int4": 0.25,  # 4-bit vs 16-bit → quarter the memory
}

_THROUGHPUT_SPEEDUP: dict[str, float] = {
    "none": 1.0,   # FP16 baseline
    "int8": 1.5,   # Approx. SmoothQuant measured improvement on A100
    "fp8":  2.0,   # H100 native FP8 GEMM acceleration
    "int4": 2.5,   # GPTQ / AWQ; higher throughput but lower quality
}

_VALID_MODES = frozenset(_MEMORY_REDUCTION.keys())


# ---------------------------------------------------------------------------
# QuantizationConfig
# ---------------------------------------------------------------------------


class QuantizationConfig:
    """
    Describes a quantization scheme and produces the vLLM kwargs needed
    to enable it.

    Parameters
    ----------
    mode:
        One of ``"none"`` (FP16 baseline), ``"int8"``, ``"fp8"``,
        or ``"int4"``.
    device:
        Informational device hint (``"auto"``, ``"h100"``, ``"a100"``).
        Does not affect the generated vLLM kwargs; useful for logging.

    Raises
    ------
    ValueError
        If ``mode`` is not one of the supported values.

    Examples
    --------
    >>> cfg = QuantizationConfig("fp8")
    >>> cfg.to_vllm_kwargs()
    {'quantization': 'fp8'}
    >>> cfg.get_theoretical_memory_reduction()
    0.5
    >>> QuantizationConfig("none").to_vllm_kwargs()
    {}
    """

    def __init__(self, mode: str = "fp8", device: str = "auto") -> None:
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Invalid quantization mode {mode!r}. "
                f"Choose from: {sorted(_VALID_MODES)}"
            )
        self.mode: str = mode
        self.device: str = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_h100_native(self) -> bool:
        """
        Return ``True`` if this configuration uses FP8 — the only mode
        with native H100 Tensor Core acceleration.

        Returns
        -------
        bool
        """
        return self.mode == "fp8"

    def to_vllm_kwargs(self) -> dict[str, Any]:
        """
        Return a dict of keyword arguments to pass to vLLM's engine or
        the ``VLLMPrefixCachedClient`` to activate this quantization.

        When mode is ``"none"`` (FP16 baseline) an empty dict is returned
        so callers do not need to special-case it.

        Returns
        -------
        dict[str, Any]
            ``{"quantization": mode}`` when mode is not ``"none"``,
            else ``{}``.
        """
        if self.mode == "none":
            return {}
        return {"quantization": self.mode}

    def get_theoretical_memory_reduction(self) -> float:
        """
        Theoretical memory footprint multiplier relative to FP16 baseline.

        A value of 0.5 means the model occupies half the GPU memory
        compared to FP16.

        Returns
        -------
        float
            Multiplier in the range (0, 1].
        """
        return _MEMORY_REDUCTION[self.mode]

    def get_theoretical_speedup(self) -> float:
        """
        Approximate token throughput multiplier relative to FP16 baseline.

        Based on published benchmarks; actual speedup depends on batch
        size, sequence length, and hardware generation.

        Returns
        -------
        float
            Multiplier >= 1.0.
        """
        return _THROUGHPUT_SPEEDUP[self.mode]

    def __repr__(self) -> str:
        return f"QuantizationConfig(mode={self.mode!r}, device={self.device!r})"


# ---------------------------------------------------------------------------
# QuantizationBenchmark
# ---------------------------------------------------------------------------


class QuantizationBenchmark:
    """
    Accumulates benchmark results for multiple quantization configurations
    and produces comparison tables relative to the FP16 (``"none"``) baseline.

    Typical usage
    -------------
    bench = QuantizationBenchmark()
    bench.record_run("none", tokens_per_second=1200, peak_memory_gb=14.2, accuracy=0.832)
    bench.record_run("fp8",  tokens_per_second=2350, peak_memory_gb=7.3,  accuracy=0.829)
    bench.record_run("int8", tokens_per_second=1780, peak_memory_gb=7.5,  accuracy=0.831)

    comparison = bench.compare()
    df_dict    = bench.to_dataframe_dict()
    """

    def __init__(self) -> None:
        # Ordered list of run records
        self._runs: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def record_run(
        self,
        config_name: str,
        tokens_per_second: float,
        peak_memory_gb: float,
        accuracy: float | None = None,
    ) -> None:
        """
        Record a benchmark measurement for one quantization configuration.

        Parameters
        ----------
        config_name:
            Label for this configuration (e.g. ``"none"``, ``"fp8"``,
            ``"int8"``, ``"int4"``).  Use ``"none"`` for the FP16 baseline.
        tokens_per_second:
            Measured decode throughput in tokens/s.
        peak_memory_gb:
            Peak GPU VRAM usage during this run (GiB).
        accuracy:
            Optional quality score (e.g. EM, F1, ROUGE-L).  Pass ``None``
            if accuracy was not measured for this run.
        """
        self._runs.append(
            {
                "config_name": config_name,
                "tokens_per_second": float(tokens_per_second),
                "peak_memory_gb": float(peak_memory_gb),
                "accuracy": float(accuracy) if accuracy is not None else None,
            }
        )

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def compare(self) -> dict[str, Any]:
        """
        Compare all recorded runs against the FP16 (``"none"``) baseline.

        The baseline is taken as the first entry whose ``config_name`` is
        ``"none"``.  If no ``"none"`` entry exists, the first recorded run
        is used as the baseline.

        Returns
        -------
        dict[str, Any]
            Structure::

                {
                    "baseline": str,          # config_name of baseline run
                    "runs": [
                        {
                            "config_name": str,
                            "tokens_per_second": float,
                            "peak_memory_gb": float,
                            "accuracy": float | None,
                            "speedup_ratio": float,        # vs baseline tps
                            "memory_reduction_ratio": float, # vs baseline memory
                            "accuracy_delta": float | None,  # vs baseline accuracy
                        },
                        ...
                    ]
                }
        """
        if not self._runs:
            return {"baseline": None, "runs": []}

        # Find baseline
        baseline = next(
            (r for r in self._runs if r["config_name"] == "none"),
            self._runs[0],
        )
        baseline_tps = baseline["tokens_per_second"]
        baseline_mem = baseline["peak_memory_gb"]
        baseline_acc = baseline["accuracy"]

        annotated_runs: list[dict[str, Any]] = []
        for run in self._runs:
            speedup = (run["tokens_per_second"] / baseline_tps) if baseline_tps > 0 else 0.0
            mem_ratio = (baseline_mem / run["peak_memory_gb"]) if run["peak_memory_gb"] > 0 else 0.0

            acc_delta: float | None = None
            if run["accuracy"] is not None and baseline_acc is not None:
                acc_delta = run["accuracy"] - baseline_acc

            annotated_runs.append(
                {
                    "config_name": run["config_name"],
                    "tokens_per_second": run["tokens_per_second"],
                    "peak_memory_gb": run["peak_memory_gb"],
                    "accuracy": run["accuracy"],
                    "speedup_ratio": round(speedup, 4),
                    "memory_reduction_ratio": round(mem_ratio, 4),
                    "accuracy_delta": round(acc_delta, 6) if acc_delta is not None else None,
                }
            )

        return {"baseline": baseline["config_name"], "runs": annotated_runs}

    def to_dataframe_dict(self) -> dict[str, list[Any]]:
        """
        Return benchmark results in a columnar format suitable for
        constructing a ``pandas.DataFrame``.

        Returns
        -------
        dict[str, list[Any]]
            Column-oriented dict.  Keys are column names; values are lists
            with one entry per recorded run (in insertion order).

            Columns: ``config_name``, ``tokens_per_second``,
            ``peak_memory_gb``, ``accuracy``.
        """
        result: dict[str, list[Any]] = {
            "config_name": [],
            "tokens_per_second": [],
            "peak_memory_gb": [],
            "accuracy": [],
        }
        for run in self._runs:
            result["config_name"].append(run["config_name"])
            result["tokens_per_second"].append(run["tokens_per_second"])
            result["peak_memory_gb"].append(run["peak_memory_gb"])
            result["accuracy"].append(run["accuracy"])
        return result

    def __repr__(self) -> str:
        return f"QuantizationBenchmark(runs={len(self._runs)})"


# ---------------------------------------------------------------------------
# GPU inspection helpers
# ---------------------------------------------------------------------------


def measure_gpu_memory_gb() -> float:
    """
    Return the current GPU memory allocated to PyTorch tensors in GiB.

    Falls back to ``0.0`` if ``torch`` is not installed or no CUDA device
    is available.

    Returns
    -------
    float
        Allocated GPU memory in gibibytes (GiB).
    """
    if not _torch_available:
        return 0.0
    try:
        if not _torch.cuda.is_available():  # type: ignore[union-attr]
            return 0.0
        bytes_allocated = _torch.cuda.memory_allocated()  # type: ignore[union-attr]
        return bytes_allocated / (1024 ** 3)
    except Exception:
        return 0.0


def get_gpu_info() -> dict[str, Any]:
    """
    Return a dictionary describing the primary CUDA GPU (device 0).

    If ``torch`` is unavailable or no GPU is present, returns a dict with
    sensible defaults (``is_h100=False``, ``is_a100=False``, etc.).

    Returns
    -------
    dict[str, Any]
        Keys:
            name (str):             Device name string (e.g. ``"NVIDIA H100 80GB HBM3"``).
            total_memory_gb (float): Total GPU VRAM in GiB.
            is_h100 (bool):         True if the device name contains "H100".
            is_a100 (bool):         True if the device name contains "A100".
    """
    default: dict[str, Any] = {
        "name": "unknown",
        "total_memory_gb": 0.0,
        "is_h100": False,
        "is_a100": False,
    }

    if not _torch_available:
        return default

    try:
        if not _torch.cuda.is_available():  # type: ignore[union-attr]
            return default

        props = _torch.cuda.get_device_properties(0)  # type: ignore[union-attr]
        name: str = props.name
        total_bytes: int = props.total_memory
        total_gb = total_bytes / (1024 ** 3)
        upper_name = name.upper()

        return {
            "name": name,
            "total_memory_gb": round(total_gb, 2),
            "is_h100": "H100" in upper_name,
            "is_a100": "A100" in upper_name,
        }
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Recommendation helper
# ---------------------------------------------------------------------------


def recommend_quantization(gpu_info: dict[str, Any]) -> QuantizationConfig:
    """
    Return the recommended :class:`QuantizationConfig` for the given GPU.

    Decision logic
    --------------
    * H100  → ``"fp8"``  (native Tensor Core support, 2x throughput)
    * A100  → ``"int8"`` (well-tested SmoothQuant / LLM.int8, no FP8 support)
    * Other → ``"none"`` (FP16 baseline; safer for older/unknown hardware)

    Parameters
    ----------
    gpu_info:
        Dict as returned by :func:`get_gpu_info`.

    Returns
    -------
    QuantizationConfig
        Recommended configuration.

    Examples
    --------
    >>> rec = recommend_quantization({"is_h100": True, "is_a100": False})
    >>> rec.mode
    'fp8'
    >>> rec = recommend_quantization({"is_h100": False, "is_a100": True})
    >>> rec.mode
    'int8'
    >>> rec = recommend_quantization({"is_h100": False, "is_a100": False})
    >>> rec.mode
    'none'
    """
    if gpu_info.get("is_h100", False):
        device_hint = "h100"
        mode = "fp8"
    elif gpu_info.get("is_a100", False):
        device_hint = "a100"
        mode = "int8"
    else:
        device_hint = gpu_info.get("name", "unknown")
        mode = "none"

    return QuantizationConfig(mode=mode, device=device_hint)
