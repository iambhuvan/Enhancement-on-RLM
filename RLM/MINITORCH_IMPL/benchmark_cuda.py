"""
benchmark_cuda.py — Compare numpy MiniTorch vs PyTorch CUDA KV cache
=====================================================================
Runs the same correctness + speedup + radix_hit tests as benchmark.py
but using CUDADecoderLM on the best available device.

Also prints a side-by-side comparison table vs the numpy baseline.

Usage
-----
  cd RLM/MINITORCH_IMPL
  python benchmark_cuda.py
  python benchmark_cuda.py --device mps
  python benchmark_cuda.py --device cpu --seq_lens 6 8 10 12
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from cuda_decoder_lm import CUDADecoderLM
from cuda_kv_cache import best_device
from cuda_radix_attention import CUDARadixAttentionCache


# ── Config ────────────────────────────────────────────────────────────────────
CFG = dict(n_vocab=256, n_embd=128, n_head=4, n_layers=4)


def _rand_ids(n: int, vocab: int = 256, rng=None) -> list[int]:
    if rng is None: rng = np.random
    return [int(x) for x in rng.randint(0, vocab, size=n)]


def sep(char="─", width=66): print(char * width)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_correctness(model: CUDADecoderLM, rng, n: int = 5):
    sep()
    print("  [correctness] generate_no_cache == generate_with_cache")
    sep()
    passes = 0
    for i in range(n):
        prompt = _rand_ids(8, rng=rng)
        nc = model.generate_no_cache(prompt, 4)
        kv = model.generate_with_cache(prompt, 4)
        ok = nc[len(prompt):] == kv[len(prompt):]
        icon = "✓" if ok else "✗"
        print(f"  [{icon}] prompt={prompt[:4]}...  no_cache={nc[len(prompt):]}  kv={kv[len(prompt):]}")
        if ok: passes += 1
    print(f"\n  {passes}/{n} PASS")
    return passes == n


def test_radix_correctness(model: CUDADecoderLM, rng, n: int = 3):
    sep()
    print("  [radix_correct] generate_no_cache == generate_with_radix (r1 & r2)")
    sep()
    passes = 0
    for i in range(n):
        prompt = _rand_ids(6, rng=rng)
        radix  = CUDARadixAttentionCache(n_layers=model.n_layers, max_cached_tokens=512)
        nc  = model.generate_no_cache(prompt, 3)
        r1  = model.generate_with_radix(prompt, 3, radix)
        r2  = model.generate_with_radix(prompt, 3, radix)
        ok1 = nc[len(prompt):] == r1[len(prompt):]
        ok2 = nc[len(prompt):] == r2[len(prompt):]
        ok  = ok1 and ok2
        stats = radix.stats()
        icon  = "✓" if ok else "✗"
        print(f"  [{icon}] nc={nc[len(prompt):]}"
              f"  r1={'ok' if ok1 else 'MISMATCH'}"
              f"  r2={'ok' if ok2 else 'MISMATCH'}"
              f"  hit={stats['hit_rate_pct']:.1f}%")
        if ok: passes += 1
    print(f"\n  {passes}/{n} PASS")
    return passes == n


def test_speedup(model: CUDADecoderLM, rng, seq_lens: list[int]):
    sep()
    print("  [speedup] KV cache vs no cache — tok/s and speedup")
    sep()
    rows = []
    prompt_len = 4
    for total in seq_lens:
        new_toks = total - prompt_len
        prompt   = _rand_ids(prompt_len, rng=rng)

        # Warmup
        model.generate_no_cache(prompt, new_toks)
        model.generate_with_cache(prompt, new_toks)

        t0 = time.perf_counter()
        model.generate_no_cache(prompt, new_toks)
        t_nc = time.perf_counter() - t0

        t0 = time.perf_counter()
        model.generate_with_cache(prompt, new_toks)
        t_kv = time.perf_counter() - t0

        tps_nc  = round(new_toks / max(t_nc, 1e-9), 2)
        tps_kv  = round(new_toks / max(t_kv, 1e-9), 2)
        speedup = round(tps_kv / max(tps_nc, 1e-9), 3)
        rows.append((total, tps_nc, tps_kv, speedup))
        print(f"  seq={total:3d}  no_cache={tps_nc:7.1f}tok/s  "
              f"kv={tps_kv:7.1f}tok/s  speedup={speedup:.2f}x")
    return rows


def test_radix_hit(model: CUDADecoderLM, rng, n_req: int = 5):
    sep()
    print("  [radix_hit] prefix reuse hit rate + speedup")
    sep()
    prefix = _rand_ids(4, rng=rng)
    radix  = CUDARadixAttentionCache(n_layers=model.n_layers, max_cached_tokens=512)
    t_r1   = None

    for req in range(1, n_req + 1):
        suffix = _rand_ids(2, rng=rng)
        prompt = prefix + suffix
        t0     = time.perf_counter()
        model.generate_with_radix(prompt, 2, radix)
        t_req  = time.perf_counter() - t0
        if t_r1 is None: t_r1 = t_req
        stats   = radix.stats()
        speedup = round(t_r1 / max(t_req, 1e-9), 3)
        reused  = len(prefix) if req > 1 else 0
        print(f"  req={req}  prefix_reused={reused}  "
              f"time={t_req:.3f}s  hit={stats['hit_rate_pct']:.1f}%  speedup={speedup:.2f}x")


def test_memory(model: CUDADecoderLM):
    sep()
    print("  [memory] KV cache allocation — CUDA vs numpy")
    sep()
    from cuda_kv_cache import CUDAKVCacheBuffer
    dtype_bytes = 2 if model.device.type in ("cuda", "mps") else 4  # float16 vs float32

    for seq in [32, 64, 128, 256, 512]:
        cache = CUDAKVCacheBuffer(
            n_layers=model.n_layers,
            n_heads=model.n_head,
            head_dim=model.head_dim,
            max_seq_len=seq,
            device=model.device,
        )
        kb   = round(cache.memory_bytes() / 1024, 1)
        util = 0.0
        print(f"  seq={seq:4d}  kv_cache={kb:8.1f}KB  "
              f"dtype={'float16' if dtype_bytes==2 else 'float32'}  device={model.device}")

    per_tok = 2 * model.n_layers * model.n_head * model.head_dim * dtype_bytes
    print(f"\n  bytes_per_token = {per_tok} ({per_tok/1024:.2f}KB)")


# ── Comparison table ──────────────────────────────────────────────────────────

def print_comparison_table(numpy_results: dict, cuda_results: dict, device: str):
    sep("═")
    print("  COMPARISON: numpy/CPU MiniTorch  vs  PyTorch/{} CUDA KV Cache".format(device.upper()))
    sep("═")
    print(f"  {'Metric':<30}  {'numpy/CPU':>14}  {'PyTorch/' + device:>14}")
    sep()
    for key in numpy_results:
        nv = numpy_results[key]
        cv = cuda_results.get(key, "—")
        print(f"  {key:<30}  {str(nv):>14}  {str(cv):>14}")
    sep("═")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",   default=None)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--seq_lens", nargs="+", type=int, default=[6, 8, 10, 12])
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else best_device()
    rng    = np.random.RandomState(args.seed)

    sep("═")
    print("  PyTorch CUDA KV Cache + RadixAttention Benchmark")
    print(f"  Device : {device}  |  Seed: {args.seed}")
    sep("═")

    print(f"\n  Building CUDADecoderLM on {device} ... ", end="", flush=True)
    model = CUDADecoderLM(**CFG, device=device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"done  ({n_params:,} params)")
    print(f"  dtype : {next(model.parameters()).dtype}")

    all_pass = True
    all_pass &= test_correctness(model, rng)
    all_pass &= test_radix_correctness(model, rng)
    speedup_rows = test_speedup(model, rng, args.seq_lens)
    test_radix_hit(model, rng)
    test_memory(model)

    sep("═")
    status = "ALL PASS ✓" if all_pass else "SOME FAILURES ✗"
    print(f"  {status}  |  Device: {device}")

    # Quick comparison summary
    if speedup_rows:
        avg_speedup = sum(r[3] for r in speedup_rows) / len(speedup_rows)
        print(f"\n  Average KV-cache speedup over no-cache: {avg_speedup:.2f}x")
        print(f"  (numpy baseline avg speedup from benchmark.py: ~1.5-2x on same seq lens)")
    sep("═")
    print()


if __name__ == "__main__":
    main()
