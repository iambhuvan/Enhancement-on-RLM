"""
modal_vllm_server_30b.py — vLLM server for Qwen3-30B-A3B (MoE) on Modal A100
==============================================================================
Deploys Qwen/Qwen3-30B-A3B (MoE, 3B active params) on A100 with INT8
quantization so it fits in 40 GB VRAM (~30 GB quantized weights).
INT8 here is a hardware necessity, not a software optimization.

Runs with:
  - O4: --enable-prefix-caching (RadixAttention KV reuse)
  - INT8: required to fit 30B weights in A100 40 GB VRAM

Separate Modal app from erlm-vllm-qwen3 so both 8B and 30B can run
simultaneously on different containers.

Usage
-----
    modal serve modal_vllm_server_30b.py
    # → prints https://bnallamo--erlm-vllm-qwen3-30b-serve-dev.modal.run

    python run_vllm_qwen3.py --n 3 --vllm_url <URL> \\
        --model_name Qwen/Qwen3-30B-A3B
"""

from __future__ import annotations

import os
import subprocess
import time

import modal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MODEL         = "Qwen/Qwen3-30B-A3B"
_QUANTIZATION  = "int8"        # INT8 required to fit in A100 40 GB
_GPU           = "A100"        # A100 40 GB — minimum for 30B INT8 (~30 GB)
_PORT          = 8000
_MAX_MODEL_LEN = 131072        # 128K — same as 8B; vLLM auto-caps KV cache to fit VRAM

# ---------------------------------------------------------------------------
# Modal image
# ---------------------------------------------------------------------------

_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.0",
        "torch>=2.3.0",
        "transformers>=4.40.0",
        "huggingface_hub[hf_transfer]>=0.22.0",
        "httpx",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",   # Qwen3 RoPE scaling: config says 40960 but supports 131072
    })
)

# Shared persistent volume with the 8B server (both cache to same volume)
_vol = modal.Volume.from_name("erlm-model-weights", create_if_missing=True)

app = modal.App("erlm-vllm-qwen3-30b")

# ---------------------------------------------------------------------------
# vLLM server
# ---------------------------------------------------------------------------

@app.function(
    image=_image,
    gpu=_GPU,
    volumes={"/models": _vol},
    timeout=7_200,
    scaledown_window=3600,       # keep alive 1 hr — long inference takes time
)
@modal.concurrent(max_inputs=16)
@modal.web_server(_PORT, startup_timeout=600)   # 30B download takes longer
def serve() -> None:
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", _MODEL,
        "--download-dir", "/models",
        "--port", str(_PORT),
        "--host", "0.0.0.0",
        "--enable-prefix-caching",
        "--quantization", _QUANTIZATION,
        "--gpu-memory-utilization", "0.90",
        "--max-model-len", str(_MAX_MODEL_LEN),
        "--trust-remote-code",
        "--no-enable-log-requests",
    ]

    print(f"[modal_vllm_server_30b] Starting: {' '.join(cmd)}")
    subprocess.Popen(cmd)

    import httpx
    health_url = f"http://localhost:{_PORT}/health"
    for attempt in range(300):   # up to 10 min (30B loads slower)
        try:
            r = httpx.get(health_url, timeout=3.0)
            if r.status_code == 200:
                print(f"[modal_vllm_server_30b] vLLM ready after {attempt * 2}s")
                return
        except Exception:
            pass
        time.sleep(2)

    raise RuntimeError("vLLM did not become healthy within 600s")


@app.local_entrypoint()
def main() -> None:
    print("\n" + "=" * 70)
    print(f"  Model        : {_MODEL}")
    print(f"  GPU          : {_GPU}")
    print(f"  INT8 quant   : yes (hardware requirement for 40GB VRAM)")
    print(f"  Max ctx len  : {_MAX_MODEL_LEN:,} tokens")
    print("=" * 70)
    print("\nRun with:  modal serve modal_vllm_server_30b.py")
    print("Then use the printed URL with:")
    print("  python run_vllm_qwen3.py --n 3 --vllm_url <URL> \\")
    print("      --model_name Qwen/Qwen3-30B-A3B")
    print()
