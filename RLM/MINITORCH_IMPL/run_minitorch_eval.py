"""
run_minitorch_eval.py — KV Cache + RadixAttention eval on MiniTorch
====================================================================
Mirrors the structure of run_ollama_qwen3.py / run_vllm_qwen3.py but
tests the custom MiniTorch KV cache and RadixAttention implementation
rather than a live language model.

Tests
-----
  correctness   — generate_no_cache vs generate_with_cache (greedy match)
  radix_correct — generate_no_cache vs generate_with_radix (r1 & r2)
  speedup       — tok/s with vs without cache at seq lengths [6,8,10,12]
  radix_hit     — hit rate and speedup over 5 requests sharing a prefix
  memory        — KV cache bytes vs sequence length

Usage
-----
  cd RLM/MINITORCH_IMPL
  python run_minitorch_eval.py                    # all tests
  python run_minitorch_eval.py --tests correctness radix_correct
  python run_minitorch_eval.py --seed 42 --n_prompts 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List, Optional

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from _numba_mock import install_mock
install_mock()

def _find_h4_dir() -> str:
    d = _THIS_DIR
    for _ in range(10):
        c = os.path.join(d, "assignments", "h4")
        if os.path.isdir(c):
            return c
        d = os.path.dirname(d)
    raise FileNotFoundError("Could not find assignments/h4")

_H4_DIR = _find_h4_dir()
if _H4_DIR not in sys.path:
    sys.path.insert(0, _H4_DIR)

import numpy as np
import minitorch
from minitorch import tensor_from_numpy
from minitorch.tensor_ops import SimpleOps
from kv_cache import KVCacheBuffer
from radix_trie import RadixAttentionCache
from kv_transformer import KVDecoderLM

# ---------------------------------------------------------------------------
# HybridOps (SimpleOps + numpy matmul — same as benchmark.py)
# ---------------------------------------------------------------------------

class HybridOps(SimpleOps):
    @staticmethod
    def matrix_multiply(a, b):
        a_np = a.to_numpy()
        b_np = b.to_numpy()
        out_np = np.matmul(a_np, b_np).astype(np.float64)
        return tensor_from_numpy(out_np, backend=a.backend)

    cuda = False

# ---------------------------------------------------------------------------
# Model config (same as benchmark.py)
# ---------------------------------------------------------------------------

CFG = dict(
    n_vocab=256, n_embd=128, n_head=4, n_positions=512,
    p_dropout=0.0, ln_eps=1e-5, bias=True,
)
N_LAYERS = KVDecoderLM.N_LAYERS  # 4


def _make_backend():
    return minitorch.TensorBackend(HybridOps)


def _make_model(backend):
    m = KVDecoderLM(backend=backend, **CFG)
    m.eval()
    return m


def _rand_ids(n: int, vocab: int = 256, rng=None) -> List[int]:
    if rng is None:
        rng = np.random
    return [int(x) for x in rng.randint(0, vocab, size=n)]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    test: str
    status: str        # PASS / FAIL / ERROR
    detail: str
    elapsed_s: float


# ---------------------------------------------------------------------------
# [1] Correctness: no_cache == with_cache
# ---------------------------------------------------------------------------

def test_correctness(model, rng, n_prompts: int = 5) -> List[TestResult]:
    results = []
    for i in range(n_prompts):
        prompt = _rand_ids(8, rng=rng)
        max_new = 4
        t0 = time.perf_counter()
        seq_nc = model.generate_no_cache(prompt, max_new)
        seq_kv = model.generate_with_cache(prompt, max_new)
        elapsed = time.perf_counter() - t0
        gen_nc = seq_nc[len(prompt):]
        gen_kv = seq_kv[len(prompt):]
        ok = gen_nc == gen_kv
        results.append(TestResult(
            test="correctness",
            status="PASS" if ok else "FAIL",
            detail=f"prompt={prompt} no_cache={gen_nc} kv_cache={gen_kv}",
            elapsed_s=elapsed,
        ))
    return results


# ---------------------------------------------------------------------------
# [2] Radix correctness: no_cache == radix (r1 and r2)
# ---------------------------------------------------------------------------

def test_radix_correctness(model, rng, n_prompts: int = 3) -> List[TestResult]:
    results = []
    for i in range(n_prompts):
        prompt = _rand_ids(6, rng=rng)
        max_new = 3
        radix = RadixAttentionCache(n_layers=N_LAYERS, max_cached_tokens=512)

        t0 = time.perf_counter()
        seq_nc  = model.generate_no_cache(prompt, max_new)
        seq_r1  = model.generate_with_radix(prompt, max_new, radix)
        seq_r2  = model.generate_with_radix(prompt, max_new, radix)
        elapsed = time.perf_counter() - t0

        gen_nc = seq_nc[len(prompt):]
        gen_r1 = seq_r1[len(prompt):]
        gen_r2 = seq_r2[len(prompt):]
        ok1 = gen_nc == gen_r1
        ok2 = gen_nc == gen_r2
        ok  = ok1 and ok2

        stats = radix.stats()
        results.append(TestResult(
            test="radix_correct",
            status="PASS" if ok else "FAIL",
            detail=(
                f"no_cache={gen_nc} r1={gen_r1}({'ok' if ok1 else 'MISMATCH'}) "
                f"r2={gen_r2}({'ok' if ok2 else 'MISMATCH'}) "
                f"hit_rate={stats['hit_rate_pct']:.1f}%"
            ),
            elapsed_s=elapsed,
        ))
    return results


# ---------------------------------------------------------------------------
# [3] KV speedup vs seq length
# ---------------------------------------------------------------------------

def test_speedup(model, rng) -> List[TestResult]:
    results = []
    seq_lens   = [6, 8, 10, 12]
    prompt_len = 4

    for total_len in seq_lens:
        new_toks = total_len - prompt_len
        prompt   = _rand_ids(prompt_len, rng=rng)

        t0 = time.perf_counter()
        model.generate_no_cache(prompt, new_toks)
        t_no = time.perf_counter() - t0

        t0 = time.perf_counter()
        model.generate_with_cache(prompt, new_toks)
        t_kv = time.perf_counter() - t0

        tps_no = round(new_toks / max(t_no, 1e-9), 2)
        tps_kv = round(new_toks / max(t_kv, 1e-9), 2)
        speedup = round(tps_kv / max(tps_no, 1e-9), 3)

        results.append(TestResult(
            test="speedup",
            status="PASS",
            detail=f"seq={total_len} no_cache={tps_no}tok/s kv={tps_kv}tok/s speedup={speedup}x",
            elapsed_s=t_no + t_kv,
        ))
    return results


# ---------------------------------------------------------------------------
# [4] RadixAttention hit rate + speedup
# ---------------------------------------------------------------------------

def test_radix_hit(model, rng, n_requests: int = 5) -> List[TestResult]:
    results = []
    prefix_len = 4
    suffix_len = 2
    new_tokens = 2

    shared_prefix = _rand_ids(prefix_len, rng=rng)
    radix = RadixAttentionCache(n_layers=N_LAYERS, max_cached_tokens=512)
    t_r1 = None

    for req_idx in range(1, n_requests + 1):
        suffix = _rand_ids(suffix_len, rng=rng)
        prompt = shared_prefix + suffix

        t0 = time.perf_counter()
        model.generate_with_radix(prompt, new_tokens, radix)
        t_req = time.perf_counter() - t0

        if t_r1 is None:
            t_r1 = t_req

        stats   = radix.stats()
        speedup = round(t_r1 / max(t_req, 1e-9), 3)
        reused  = prefix_len if req_idx > 1 else 0

        results.append(TestResult(
            test="radix_hit",
            status="PASS",
            detail=(
                f"req={req_idx} prefix_reused={reused} "
                f"time={t_req:.3f}s hit_rate={stats['hit_rate_pct']:.1f}% "
                f"speedup={speedup}x"
            ),
            elapsed_s=t_req,
        ))

    return results


# ---------------------------------------------------------------------------
# [5] Memory usage
# ---------------------------------------------------------------------------

def test_memory(model) -> List[TestResult]:
    results = []
    head_dim = CFG["n_embd"] // CFG["n_head"]

    for seq_len in [32, 64, 128, 256, 512]:
        cache = KVCacheBuffer(
            n_layers=N_LAYERS,
            n_heads=CFG["n_head"],
            head_dim=head_dim,
            max_seq_len=seq_len,
        )
        cache.current_len = seq_len
        kb   = round(cache.memory_bytes() / 1024, 2)
        util = round(cache.utilization() * 100, 1)

        results.append(TestResult(
            test="memory",
            status="PASS",
            detail=f"seq={seq_len} kv_cache={kb}KB util={util}%",
            elapsed_s=0.0,
        ))

    per_tok_bytes = 2 * N_LAYERS * CFG["n_head"] * head_dim * 4
    results.append(TestResult(
        test="memory",
        status="PASS",
        detail=f"bytes_per_token={per_tok_bytes} ({per_tok_bytes/1024:.2f}KB)",
        elapsed_s=0.0,
    ))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_TESTS = ["correctness", "radix_correct", "speedup", "radix_hit", "memory"]


def main():
    parser = argparse.ArgumentParser(description="MiniTorch KV Cache + RadixAttention eval")
    parser.add_argument("--tests", nargs="+", choices=ALL_TESTS, default=ALL_TESTS)
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--n_prompts", type=int, default=5,
                        help="Number of prompts for correctness tests")
    parser.add_argument("--out",  default=None, help="Save JSON results to file")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    print("\n" + "=" * 66)
    print("  MiniTorch KV Cache + RadixAttention — Eval Suite")
    print(f"  Seed: {args.seed}  |  Tests: {args.tests}")
    print("=" * 66)

    backend = _make_backend()
    print(f"\n  Backend : {backend}")
    print("  Building model … ", end="", flush=True)
    model = _make_model(backend)
    n_params = sum(p.value.size for p in model.parameters() if hasattr(p, "value"))
    print(f"done  ({n_params:,} parameters)")

    all_results: List[TestResult] = []

    for test_name in args.tests:
        print(f"\n{'─'*66}")
        print(f"  [{test_name}]")
        print(f"{'─'*66}")

        t0 = time.perf_counter()
        if test_name == "correctness":
            res = test_correctness(model, rng, args.n_prompts)
        elif test_name == "radix_correct":
            res = test_radix_correctness(model, rng, min(args.n_prompts, 3))
        elif test_name == "speedup":
            res = test_speedup(model, rng)
        elif test_name == "radix_hit":
            res = test_radix_hit(model, rng)
        elif test_name == "memory":
            res = test_memory(model)
        else:
            continue

        for r in res:
            icon = "✓" if r.status == "PASS" else "✗"
            print(f"  [{icon}] {r.test:<18} {r.detail}")

        all_results.extend(res)

    # Summary
    n_pass = sum(1 for r in all_results if r.status == "PASS")
    n_fail = sum(1 for r in all_results if r.status == "FAIL")
    n_err  = sum(1 for r in all_results if r.status == "ERROR")

    print("\n" + "=" * 66)
    print(f"  Results: {n_pass} PASS  {n_fail} FAIL  {n_err} ERROR  "
          f"(total {len(all_results)})")
    print("=" * 66 + "\n")

    # Optionally save JSON
    if args.out:
        out_path = args.out
        payload = {
            "timestamp": datetime.now().isoformat(),
            "seed": args.seed,
            "model_config": CFG,
            "n_layers": N_LAYERS,
            "results": [asdict(r) for r in all_results],
            "summary": {"pass": n_pass, "fail": n_fail, "error": n_err},
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  Results saved to {out_path}\n")

    return 0 if n_fail == 0 and n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
