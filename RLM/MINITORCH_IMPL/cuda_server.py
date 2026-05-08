"""
cuda_server.py — OpenAI-compatible HTTP server backed by CUDADecoderLM
=======================================================================
Exposes POST /v1/chat/completions exactly like Ollama / vLLM.
Point any OpenAI client (including ERLM's OllamaClient) at:
    http://localhost:8003/v1

Usage
-----
  # Start server
  cd RLM/MINITORCH_IMPL
  python cuda_server.py [--port 8003] [--radix] [--device cuda|mps|cpu]

  # Point ERLM at it
  python ERLM/EVALS/run_ollama_qwen3.py \
      --base_url http://localhost:8003/v1 \
      --model cuda-decoder \
      --n 3 --methods rlm_baseline erlm_o1o2

Notes
-----
- Model is a toy (n_vocab=256, n_embd=128) — outputs are random bytes,
  NOT real language. This demo proves the infrastructure works end-to-end.
- To use real GPT-2 weights, pass --gpt2 flag (requires transformers).
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch

from cuda_decoder_lm import CUDADecoderLM
from cuda_kv_cache import best_device
from cuda_radix_attention import CUDARadixAttentionCache


# ── Global model (loaded once at startup) ────────────────────────────────────
_model: CUDADecoderLM | None = None
_radix_cache: CUDARadixAttentionCache | None = None
_use_radix: bool = False


def _encode(text: str, max_len: int = 16) -> list[int]:
    """Crude char → token mapping: ord(c) % n_vocab."""
    vocab = _model.n_vocab if _model else 256
    return [ord(c) % vocab for c in text[:max_len]]


def _decode(token_ids: list[int]) -> str:
    """Map token ids back to printable chars."""
    return "".join(chr(max(32, min(126, t))) for t in token_ids)


def _run_completion(prompt_text: str, max_tokens: int = 20) -> dict:
    """Run generation and return an OpenAI-compatible response dict."""
    prompt_ids = _encode(prompt_text)
    t0 = time.perf_counter()

    if _use_radix and _radix_cache is not None:
        output_ids = _model.generate_with_radix(prompt_ids, max_tokens, _radix_cache)
    else:
        output_ids = _model.generate_with_cache(prompt_ids, max_tokens)

    elapsed = time.perf_counter() - t0
    new_ids  = output_ids[len(prompt_ids):]
    content  = _decode(new_ids)

    return {
        "id":      f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   "cuda-decoder",
        "choices": [{
            "index":   0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens":     len(prompt_ids),
            "completion_tokens": len(new_ids),
            "total_tokens":      len(prompt_ids) + len(new_ids),
        },
        "_elapsed_s": round(elapsed, 3),
    }


# ── HTTP Handler ──────────────────────────────────────────────────────────────

class OpenAIHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass   # suppress default access log

    def _send_json(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            device = str(_model.device) if _model else "none"
            self._send_json(200, {"status": "ok", "device": device,
                                  "radix": _use_radix})
        elif self.path == "/v1/models":
            self._send_json(200, {"object": "list", "data": [
                {"id": "cuda-decoder", "object": "model"}
            ]})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        body   = json.loads(self.rfile.read(length))

        messages   = body.get("messages", [])
        max_tokens = body.get("max_tokens", 20)

        # Extract last user message
        prompt_text = ""
        for msg in reversed(messages):
            if msg.get("role") in ("user", "system"):
                prompt_text = msg.get("content", "")
                break

        try:
            result = _run_completion(prompt_text, max_tokens=min(max_tokens, 64))
            self._send_json(200, result)
        except Exception as e:
            self._send_json(500, {"error": str(e)})


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global _model, _radix_cache, _use_radix

    parser = argparse.ArgumentParser(description="CUDA Decoder LM OpenAI-compatible server")
    parser.add_argument("--port",    type=int,  default=8003)
    parser.add_argument("--radix",   action="store_true", help="Enable RadixAttention cache")
    parser.add_argument("--device",  default=None, help="cuda|mps|cpu (auto-detect if omitted)")
    parser.add_argument("--n_embd",  type=int, default=128)
    parser.add_argument("--n_head",  type=int, default=4)
    parser.add_argument("--n_layers",type=int, default=4)
    parser.add_argument("--n_vocab", type=int, default=256)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else best_device()
    print(f"\n  CUDA Decoder LM Server")
    print(f"  Device  : {device}")
    print(f"  Config  : vocab={args.n_vocab} embd={args.n_embd} heads={args.n_head} layers={args.n_layers}")

    _model = CUDADecoderLM(
        n_vocab=args.n_vocab,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layers=args.n_layers,
        device=device,
    )
    _model.eval()
    n_params = sum(p.numel() for p in _model.parameters())
    print(f"  Params  : {n_params:,}")

    _use_radix = args.radix
    if _use_radix:
        _radix_cache = CUDARadixAttentionCache(
            n_layers=args.n_layers,
            max_cached_tokens=2048,
            pin_memory=(device.type == "cuda"),
        )
        print(f"  RadixAttention: enabled")
    else:
        print(f"  RadixAttention: disabled (--radix to enable)")

    print(f"\n  Listening on http://localhost:{args.port}/v1")
    print(f"  Health:  GET  http://localhost:{args.port}/health")
    print(f"  Chat:    POST http://localhost:{args.port}/v1/chat/completions\n")

    server = HTTPServer(("0.0.0.0", args.port), OpenAIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")


if __name__ == "__main__":
    main()
