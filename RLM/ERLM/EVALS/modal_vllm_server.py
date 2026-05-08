"""
modal_vllm_server.py — vLLM server on Modal GPU for O4 benchmarking
=====================================================================
Deploys Qwen/Qwen3-8B on a Modal GPU with:
  - O4: --enable-prefix-caching (RadixAttention KV reuse)

Exposes an OpenAI-compatible endpoint that run_vllm_qwen3.py connects to.

Usage
-----
    # Deploy (keeps running, prints the endpoint URL):
    modal serve modal_vllm_server.py

    # One-shot run (terminates after idle):
    modal run modal_vllm_server.py

GPU options (set GPU env var or edit _GPU below):
    A10G  — 24 GB VRAM, ~$0.50/hr
    A100  — 40 GB VRAM, ~$2.80/hr
    H100  — 80 GB VRAM, ~$4.50/hr

After `modal serve` prints the URL, pass it to run_vllm_qwen3.py:
    python run_vllm_qwen3.py --n 50 --vllm_url https://YOUR-MODAL-URL
"""

from __future__ import annotations

import os
import subprocess
import time

import modal

# ---------------------------------------------------------------------------
# Configuration — edit these or override via environment variables
# ---------------------------------------------------------------------------

_MODEL        = os.environ.get("MODEL", "Qwen/Qwen3-8B")
_GPU          = os.environ.get("GPU", "A100")            # A10G=24GB (too small), A100=40GB, H100=80GB
_PORT         = 8000
_MAX_MODEL_LEN = 32768    # 32K — fits in A100's KV cache budget after model weights

# ---------------------------------------------------------------------------
# Modal image — vLLM + HF transfer
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

# Persistent volume — caches model weights across runs (avoids re-downloading 16 GB)
_vol = modal.Volume.from_name("erlm-model-weights", create_if_missing=True)

app = modal.App("erlm-vllm-qwen3")

# ---------------------------------------------------------------------------
# vLLM server function
# ---------------------------------------------------------------------------

@app.function(
    image=_image,
    gpu=_GPU,
    volumes={"/models": _vol},
    timeout=7_200,               # 2 hr max session
    scaledown_window=3600,       # keep alive 1 hr — long inference takes time
)
@modal.concurrent(max_inputs=32)
@modal.web_server(_PORT, startup_timeout=300)
def serve() -> None:
    """
    Start vLLM's OpenAI-compatible server inside the Modal container.
    Modal's @web_server decorator tunnels :PORT to a public HTTPS endpoint.
    """
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", _MODEL,
        "--download-dir", "/models",
        "--port", str(_PORT),
        "--host", "0.0.0.0",
        "--enable-prefix-caching",         # O4: RadixAttention
        "--gpu-memory-utilization", "0.85",
        "--max-model-len", str(_MAX_MODEL_LEN),
        "--trust-remote-code",
        "--no-enable-log-requests",
    ]

    print(f"[modal_vllm_server] Starting: {' '.join(cmd)}")
    subprocess.Popen(cmd)

    # Wait for vLLM to be ready (model load takes ~60-120s on first run)
    import httpx
    health_url = f"http://localhost:{_PORT}/health"
    for attempt in range(150):   # up to 5 min
        try:
            r = httpx.get(health_url, timeout=3.0)
            if r.status_code == 200:
                print(f"[modal_vllm_server] vLLM ready after {attempt * 2}s")
                return
        except Exception:
            pass
        time.sleep(2)

    raise RuntimeError("vLLM did not become healthy within 300s")


# ---------------------------------------------------------------------------
# Local entrypoint — prints the URL for the caller
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main() -> None:
    print("\n" + "=" * 70)
    print(f"  Model        : {_MODEL}")
    print(f"  GPU          : {_GPU}")
    print(f"  Max ctx len  : {_MAX_MODEL_LEN:,} tokens")
    print("=" * 70)
    print("\nRun with:  modal serve modal_vllm_server.py")
    print("Then use the printed URL with:")
    print("  python run_vllm_qwen3.py --n 50 --vllm_url <URL>")
    print()
