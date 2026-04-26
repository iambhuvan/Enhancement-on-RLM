"""
run_vllm_qwen3.py — LongBench-v2 CodeQA eval using Qwen3-8B via vLLM (Modal)
==============================================================================
Evaluates four methods on LongBench-v2 CodeQA samples using Qwen3-8B
served by a vLLM endpoint (e.g. deployed via modal_vllm_server.py).

Adds O4 (KV prefix cache) and O5 (quantization) on top of O1-O3.
Tracks additional metrics vs run_ollama_qwen3.py:
  - Time to First Token (TTFT, ms)
  - KV cache hit rate (% from vLLM Prometheus /metrics)
  - GPU memory usage (GB, if torch available in vLLM container)
  - Tokens/sec speedup over Ollama baseline

Methods
-------
  base_model     — Direct vLLM call, doc truncated to 450K chars
  rlm_baseline   — RLM iterative REPL loop via vLLM
  erlm_o1o2      — ERLM with O1 (TF-IDF) + O2 (budget) via vLLM
  erlm_o1o2o3    — ERLM with O1 + O2 + O3 (async) via vLLM
  All methods benefit from O4 (prefix cache) and O5 (quantization) server-side.

Requirements
------------
  modal serve modal_vllm_server.py   # keep running in a separate terminal
  pip install openai httpx

Usage
-----
    python run_vllm_qwen3.py --n 50 --vllm_url https://YOUR-MODAL-URL
    python run_vllm_qwen3.py --n 3  --vllm_url https://YOUR-MODAL-URL --quantization int8
    python run_vllm_qwen3.py --n 50 --vllm_url https://YOUR-MODAL-URL --methods erlm_o1o2o3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_ERLM_DIR     = os.path.join(_THIS_DIR, "..")
_BASELINE_DIR = os.path.join(_ERLM_DIR, "..", "BASELINE")

sys.path.insert(0, _BASELINE_DIR)
sys.path.insert(0, _ERLM_DIR)
sys.path.insert(0, _THIS_DIR)

# ---------------------------------------------------------------------------
# Model constants — Qwen3-8B via vLLM (Modal)
# ---------------------------------------------------------------------------

_HF_MODEL_NAME  = "Qwen/Qwen3-8B"
_MODEL_LABEL    = "vllm_qwen3_8b"

# Same doc window as Ollama Qwen3 for fair comparison
_BASE_MODEL_DOC_CHARS = 450_000
_MAX_ITERATIONS = 3
_MAX_TIMEOUT_S  = 600.0

# Same sample selection as other eval scripts
_MIN_DOC_LENGTH = 800_000
_MAX_DOC_LENGTH = 4_000_000

# Ollama baseline tok/s for speedup calculation (from sanity runs)
_OLLAMA_BASELINE_TOKPS = 115.0

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(log_dir: str, ts: str, label: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{label}_{ts}.log")
    logger = logging.getLogger(label)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log: {log_path}")
    return logger

log = logging.getLogger(_MODEL_LABEL)

# ---------------------------------------------------------------------------
# Result dataclass — extends base with O4/O5 metrics
# ---------------------------------------------------------------------------

@dataclass
class SampleResult:
    model:               str
    dataset:             str
    sample_id:           str
    method:              str
    quantization:        str
    prediction:          str
    ground_truth:        str
    exact_match:         float
    f1:                  float
    tokens_used:         int
    input_tokens:        int
    output_tokens:       int
    tokens_per_sec:      float
    speedup_vs_ollama:   float
    ttft_ms:             float     # O4: time to first token
    kv_cache_hit_rate:   float     # O4: prefix cache hit %
    gpu_memory_gb:       float     # O5: VRAM used
    iterations:          int
    wall_clock_s:        float
    error:               str   = ""
    o1_chunks:           int   = 0
    early_terminated:    bool  = False
    termination_reason:  str   = ""
    o3_speedup_ratio:    float = 0.0
    o3_parallel_batches: int   = 0

# ---------------------------------------------------------------------------
# Metrics (identical extraction logic to other eval scripts)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    import re, string
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())

def _extract_mc_letter(text: str) -> str:
    import re
    m = re.search(r'(?:^|[\s:()\[\].,"\'`])\b([A-Da-d])\b(?:[\s:).,"\'`]|$)', text)
    if m:
        return m.group(1).lower()
    return _normalize(text)

def _extract_final_answer(text: str) -> str:
    """Extract MC letter from verbose REPL output before scoring."""
    import re
    # Strip <think>...</think> blocks (Qwen3 thinking mode)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # FINAL("C") / FINAL('c') / FINAL(C) — single letter
    m = re.search(r'FINAL\s*\(\s*["\']?\s*([A-Da-d])\s*["\']?\s*\)', text)
    if m:
        return m.group(1)
    # "the answer is C" / "answer: C" / "Answer (C)"
    m = re.search(r'(?:answer(?:s)?(?:\s+is|\s*:|\s+option|\s+choice)?\s*[\(\["\'\s]*)([A-Da-d])\b', text, re.IGNORECASE)
    if m:
        return m.group(1)
    # "correct answer is C" / "C is correct"
    m = re.search(r'\b([A-Da-d])\b\s+is\s+(?:the\s+)?correct', text, re.IGNORECASE)
    if m:
        return m.group(1)
    # **C** bold markdown
    m = re.search(r'\*\*([A-Da-d])\*\*', text)
    if m:
        return m.group(1)
    # Standalone letter at end of text
    m = re.search(r'\b([A-Da-d])[.\s]*$', text.strip())
    if m:
        return m.group(1)
    # FINAL("some longer text...") — scan inner text for a letter
    m = re.search(r'FINAL\s*\(\s*["\']([^"\']{0,500})["\']', text, re.DOTALL)
    if m:
        inner = m.group(1)
        lm = re.search(r'\b([A-Da-d])\b', inner)
        if lm:
            return lm.group(1)
    return text

def exact_match(pred: str, gold: str) -> float:
    pred = _extract_final_answer(pred)
    gold_n = _normalize(gold)
    if len(gold_n) == 1 and gold_n in "abcd":
        return 1.0 if _extract_mc_letter(pred) == gold_n else 0.0
    return 1.0 if _normalize(pred) == gold_n else 0.0

def f1_score(pred: str, gold: str) -> float:
    from collections import Counter
    pred = _extract_final_answer(pred)
    gold_n = _normalize(gold)
    if len(gold_n) == 1 and gold_n in "abcd":
        return 1.0 if _extract_mc_letter(pred) == gold_n else 0.0
    p_toks = _normalize(pred).split()
    g_toks = gold_n.split()
    if not p_toks or not g_toks:
        return 0.0
    common = sum((Counter(p_toks) & Counter(g_toks)).values())
    if common == 0:
        return 0.0
    precision = common / len(p_toks)
    recall    = common / len(g_toks)
    return 2 * precision * recall / (precision + recall)

# ---------------------------------------------------------------------------
# vLLM client factory
# ---------------------------------------------------------------------------

def _make_vllm_client(vllm_url: str, quantization: str) -> Any:
    """Create a VLLMPrefixCachedClient pointing to the Modal endpoint."""
    sys.path.insert(0, os.path.join(_ERLM_DIR, "optimisations"))
    from kv_prefix_cache import VLLMPrefixCachedClient
    base_url = vllm_url.rstrip("/") + "/v1"
    return VLLMPrefixCachedClient(
        model_name=_HF_MODEL_NAME,
        base_url=base_url,
        api_key="EMPTY",
        enable_prefix_caching=True,
        quantization=quantization if quantization != "none" else None,
        max_tokens=4096,
        temperature=0.0,
        timeout=300.0,
    )

def _get_kv_cache_metrics(vllm_url: str) -> dict[str, float]:
    """Scrape vLLM Prometheus metrics from the Modal endpoint."""
    try:
        import httpx
        metrics_url = vllm_url.rstrip("/") + "/metrics"
        r = httpx.get(metrics_url, timeout=5.0)
        if r.status_code != 200:
            return {}
        text = r.text
        result = {}
        for name in ["vllm:gpu_prefix_cache_hit_rate", "vllm:gpu_cache_usage_perc",
                     "vllm:avg_prompt_throughput_toks_per_s"]:
            for line in text.splitlines():
                if line.strip().startswith(name) and not line.startswith("#"):
                    parts = line.strip().split()
                    if parts:
                        try:
                            result[name] = float(parts[-1])
                        except ValueError:
                            pass
                    break
        return result
    except Exception:
        return {}

def _make_backend_kwargs(vllm_url: str) -> dict[str, Any]:
    return {
        "model_name": _HF_MODEL_NAME,
        "base_url":   vllm_url.rstrip("/") + "/v1",
        "api_key":    "EMPTY",
    }

# ---------------------------------------------------------------------------
# Base-model wrapper (direct vLLM call, no RLM loop)
# ---------------------------------------------------------------------------

class _BaseModelWrapper:
    def __init__(self, vllm_client: Any) -> None:
        self._client = vllm_client

    def completion(self, document: str, question: str) -> Any:
        from rlm.core.types import RLMChatCompletion, UsageSummary
        truncated  = document[:_BASE_MODEL_DOC_CHARS]
        trunc_note = (
            f"\n\n[Document truncated to {_BASE_MODEL_DOC_CHARS:,} chars "
            f"of {len(document):,} total — context window limit.]"
            if len(document) > _BASE_MODEL_DOC_CHARS else ""
        )
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the question solely based on "
                    "the provided document. For multiple-choice questions respond with "
                    "only the letter (A, B, C, or D)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Document:\n{truncated}{trunc_note}\n\n"
                    f"Question: {question}\n\nAnswer (letter only):"
                ),
            },
        ]
        t0 = time.perf_counter()
        response_text = self._client.completion(prompt)
        elapsed = time.perf_counter() - t0
        usage = self._client.get_usage_summary() if hasattr(self._client, "get_usage_summary") else None
        return RLMChatCompletion(
            root_model=_HF_MODEL_NAME, prompt=prompt, response=response_text,
            usage_summary=usage or UsageSummary(model_usage={}),
            execution_time=elapsed,
        )

# ---------------------------------------------------------------------------
# RLM / ERLM builders (using ollama backend pointed at vLLM's OpenAI API)
# ---------------------------------------------------------------------------

def build_base_model(vllm_client: Any) -> _BaseModelWrapper:
    return _BaseModelWrapper(vllm_client)

def build_rlm_baseline(bkw: dict[str, Any]) -> Any:
    from rlm.core.rlm import RLM
    _iters = [0]
    m = RLM(
        backend="ollama", backend_kwargs=bkw,
        max_depth=1, max_iterations=_MAX_ITERATIONS, max_timeout=_MAX_TIMEOUT_S,
        on_iteration_complete=lambda i, d, e: _iters.__setitem__(0, i + 1),
    )
    m._erlm_iter_count = _iters
    return m

def build_erlm(bkw: dict[str, Any], o1: bool, o2: bool, o3: bool) -> Any:
    from erlm import EnhancedRLM
    _iters = [0]
    m = EnhancedRLM(
        backend="ollama", backend_kwargs=bkw,
        max_depth=1, max_iterations=_MAX_ITERATIONS, max_timeout=_MAX_TIMEOUT_S,
        max_tokens=50_000,           # O2 budget (token-accurate with vLLM)
        enable_indexing=o1, enable_budget=o2, enable_async=o3,
        enable_kv_cache=True,        # O4: flag for metadata tracking
        enable_fp8=False,            # O5: handled server-side by vLLM
        indexer_chunk_size=1000,
        indexer_overlap=100,
        indexer_top_k=3,
        on_iteration_complete=lambda i, d, e: _iters.__setitem__(0, i + 1),
    )
    m._erlm_iter_count = _iters
    return m

# ---------------------------------------------------------------------------
# Run one sample
# ---------------------------------------------------------------------------

def run_sample(
    model_fn, document: str, question: str,
    gold: str, sample_id: str, method: str,
    vllm_url: str, quantization: str, vllm_client: Any,
) -> SampleResult:
    t0 = time.perf_counter()
    prediction = ""; tokens_used = 0; input_tokens = 0; output_tokens = 0
    tokens_per_sec = 0.0; iterations = 0; error = ""
    o1_chunks = 0; early_terminated = False; termination_reason = ""
    o3_speedup_ratio = 0.0; o3_parallel_batches = 0

    # O4 metrics — snapshot before and after call
    kv_before = _get_kv_cache_metrics(vllm_url)

    try:
        model = model_fn()
        if isinstance(model, _BaseModelWrapper):
            result = model.completion(document, question)
        else:
            _mc_hint = "\n\nIMPORTANT: Your final answer must be a single letter only: A, B, C, or D."
            result = model.completion(document, root_prompt=question + _mc_hint)

        prediction = (
            getattr(result, "response", None)
            or getattr(result, "final_answer", None) or ""
        )
        us = getattr(result, "usage_summary", None) or getattr(result, "usage", None)
        if us is not None:
            input_tokens  = getattr(us, "total_input_tokens", 0)
            output_tokens = getattr(us, "total_output_tokens", 0)
            tokens_used   = input_tokens + output_tokens

        if hasattr(model, '_erlm_iter_count'):
            iterations = model._erlm_iter_count[0]
        elif hasattr(result, "metadata") and isinstance(result.metadata, dict):
            iters_list = result.metadata.get("iterations", [])
            iterations = len(iters_list) if isinstance(iters_list, list) else 0

        if hasattr(model, "_indexer") and model._indexer is not None:
            o1_chunks = len(model._indexer.chunk_offsets)
        if hasattr(model, "_budget_controller") and model._budget_controller is not None:
            reason = model._budget_controller.termination_reason or ""
            termination_reason = reason; early_terminated = bool(reason)
        if hasattr(model, "_async_manager") and model._async_manager is not None:
            stats = model._async_manager.get_speedup_stats()
            o3_speedup_ratio    = stats.get("speedup_ratio", 0.0)
            o3_parallel_batches = stats.get("total_parallel_batches", 0)

        log.info(f"  [pred] gold={gold!r}  pred={prediction[:200]!r}")
        log.info(
            f"  [tok]  total={tokens_used}  in={input_tokens}  out={output_tokens}  "
            f"chunks={o1_chunks}  early_stop={early_terminated}"
        )

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        log.error(f"[{method}] sample={sample_id} FAILED: {error}")
        log.debug(traceback.format_exc())

    wall = time.perf_counter() - t0

    # O4 metrics — snapshot after call
    kv_after = _get_kv_cache_metrics(vllm_url)
    kv_hit_rate = kv_after.get("vllm:gpu_prefix_cache_hit_rate", 0.0) * 100.0

    # TTFT from vLLM client
    ttft_ms = 0.0
    if hasattr(vllm_client, "avg_ttft"):
        ttft_ms = vllm_client.avg_ttft() * 1000.0

    # O5 — GPU memory from nvidia-smi via vLLM metrics
    gpu_memory_gb = 0.0
    if kv_after.get("vllm:gpu_cache_usage_perc", 0.0) > 0:
        # vLLM doesn't expose raw GB directly; estimate from cache usage %
        gpu_memory_gb = -1.0   # flag as "measured server-side"

    em  = exact_match(prediction, gold)
    f1  = f1_score(prediction, gold)
    tokens_per_sec = round(tokens_used / wall, 1) if wall > 0 else 0.0
    speedup = round(tokens_per_sec / _OLLAMA_BASELINE_TOKPS, 2) if _OLLAMA_BASELINE_TOKPS > 0 else 0.0

    return SampleResult(
        model=_HF_MODEL_NAME, dataset="codeqa", sample_id=sample_id, method=method,
        quantization=quantization,
        prediction=prediction[:500], ground_truth=gold,
        exact_match=em, f1=f1,
        tokens_used=tokens_used, input_tokens=input_tokens, output_tokens=output_tokens,
        tokens_per_sec=tokens_per_sec, speedup_vs_ollama=speedup,
        ttft_ms=round(ttft_ms, 1), kv_cache_hit_rate=round(kv_hit_rate, 1),
        gpu_memory_gb=gpu_memory_gb,
        iterations=iterations, wall_clock_s=round(wall, 2), error=error,
        o1_chunks=o1_chunks, early_terminated=early_terminated,
        termination_reason=termination_reason, o3_speedup_ratio=o3_speedup_ratio,
        o3_parallel_batches=o3_parallel_batches,
    )

# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def _print_summary(results: list[SampleResult], methods: list[str], quantization: str) -> None:
    from collections import defaultdict
    import statistics

    log.info("\n" + "=" * 120)
    log.info(f"SUMMARY — {_HF_MODEL_NAME} via vLLM | quant={quantization} | CodeQA / LongBench-v2")
    log.info("=" * 120)

    by_method: dict[str, list[SampleResult]] = defaultdict(list)
    for r in results:
        by_method[r.method].append(r)

    log.info("\n── Table 1: Quality & Token Efficiency ──")
    h1 = (f"{'Method':<20} {'N':>3} {'OK':>3} {'EM':>6} {'F1':>6} "
          f"{'TotalTok':>9} {'Tok/s':>7} {'SpeedupVsOllama':>16} {'Time(s)':>8}")
    log.info(h1); log.info("-" * len(h1))
    for method in methods:
        rows = by_method.get(method, [])
        if not rows: continue
        ok = [r for r in rows if not r.error]
        if not ok:
            log.info(f"{method:<20} {len(rows):>3} {'0':>3}  — all errored"); continue
        em  = statistics.mean(r.exact_match for r in ok)
        f1  = statistics.mean(r.f1 for r in ok)
        tok = statistics.mean(r.tokens_used for r in ok)
        tps = statistics.mean(r.tokens_per_sec for r in ok)
        spd = statistics.mean(r.speedup_vs_ollama for r in ok)
        wt  = statistics.mean(r.wall_clock_s for r in rows)
        log.info(f"{method:<20} {len(rows):>3} {len(ok):>3} {em:>6.3f} {f1:>6.3f} "
                 f"{tok:>9.0f} {tps:>7.1f} {spd:>16.2f}x {wt:>8.1f}")
    log.info("-" * len(h1))

    log.info("\n── Table 2: O4/O5 System Metrics ──")
    h2 = (f"{'Method':<20} {'TTFT(ms)':>9} {'KV-Hit%':>8} {'O1Chunks':>9} "
          f"{'TokRedux%':>10} {'O3Speedup':>10}")
    log.info(h2); log.info("-" * len(h2))
    baseline_tok = None
    for method in methods:
        rows = by_method.get(method, [])
        if not rows: continue
        ok = [r for r in rows if not r.error]
        if not ok: continue
        if method == "rlm_baseline":
            baseline_tok = statistics.mean(r.tokens_used for r in ok)
        tok     = statistics.mean(r.tokens_used for r in ok)
        ttft    = statistics.mean(r.ttft_ms for r in ok)
        kv_hit  = statistics.mean(r.kv_cache_hit_rate for r in ok)
        chunks  = statistics.mean(r.o1_chunks for r in ok)
        o3spd   = statistics.mean(r.o3_speedup_ratio for r in ok)
        tok_red = (1 - tok / baseline_tok) * 100 if baseline_tok and baseline_tok > 0 else 0.0
        log.info(f"{method:<20} {ttft:>9.1f} {kv_hit:>8.1f} {chunks:>9.0f} "
                 f"{tok_red:>10.1f}% {o3spd:>10.2f}x")
    log.info("-" * len(h2))


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_samples(n: int, min_chars: int, max_chars: int, seed: int) -> list[dict]:
    from benchmarks.longbench_codeqa import load_longbench_codeqa
    import random
    all_samples = load_longbench_codeqa()
    filtered = [s for s in all_samples
                if min_chars <= len(s.get("context", "")) <= max_chars]
    rng = random.Random(seed)
    rng.shuffle(filtered)
    return filtered[:n]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ERLM O4/O5 eval via vLLM on Modal")
    p.add_argument("--n", type=int, default=50, help="Number of samples")
    p.add_argument("--vllm_url", required=True,
                   help="Modal vLLM endpoint URL (from `modal serve modal_vllm_server.py`)")
    p.add_argument("--quantization", default="none", choices=["none", "int8", "fp8"],
                   help="Quantization mode the vLLM server was started with (for logging)")
    p.add_argument("--methods", nargs="+",
                   default=["base_model", "rlm_baseline", "erlm_o1o2", "erlm_o1o2o3"],
                   choices=["base_model", "rlm_baseline", "erlm_o1o2", "erlm_o1o2o3"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="results/vllm_qwen3_8b")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    quant = args.quantization
    label = f"{_MODEL_LABEL}_{quant}" if quant != "none" else _MODEL_LABEL
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    global log
    log = _setup_logging(args.out_dir, ts, label)

    log.info("=" * 70)
    log.info(f"Model          : {_HF_MODEL_NAME}")
    log.info(f"Backend        : vLLM on Modal ({args.vllm_url})")
    log.info(f"Quantization   : {quant}")
    log.info(f"O4 KV Cache    : enabled (server-side via --enable-prefix-caching)")
    log.info(f"Context window : 128K tokens ≈ 480K chars")
    log.info(f"Base-model cap : {_BASE_MODEL_DOC_CHARS:,} chars")
    log.info(f"Samples        : {args.n}")
    log.info(f"Methods        : {args.methods}")
    log.info(f"Doc length     : {_MIN_DOC_LENGTH:,} – {_MAX_DOC_LENGTH:,} chars")
    log.info(f"Max iterations : {_MAX_ITERATIONS}")
    log.info(f"Timeout/sample : {_MAX_TIMEOUT_S}s")
    log.info(f"Output dir     : {args.out_dir}")
    log.info("=" * 70)

    # Shared vLLM client (tracks TTFT across all calls)
    vllm_client = _make_vllm_client(args.vllm_url, quant)
    bkw = _make_backend_kwargs(args.vllm_url)

    samples = _load_samples(args.n, _MIN_DOC_LENGTH, _MAX_DOC_LENGTH, args.seed)
    log.info(f"Loaded {len(samples)} samples. Running {len(args.methods)} methods…")

    results: list[SampleResult] = []

    for i, sample in enumerate(samples, 1):
        doc       = sample.get("context", "")
        question  = sample.get("input", "")
        gold      = sample.get("answers", [""])[0] if isinstance(sample.get("answers"), list) else sample.get("answers", "")
        sample_id = sample.get("_id", f"sample_{i}")

        log.info(f"[codeqa] Sample {i}/{len(samples)} id={sample_id}  "
                 f"doc_chars={len(doc):,}  q={question[:80]!r}")

        method_fns = {
            "base_model":    lambda: build_base_model(vllm_client),
            "rlm_baseline":  lambda: build_rlm_baseline(bkw),
            "erlm_o1o2":     lambda: build_erlm(bkw, o1=True, o2=True, o3=False),
            "erlm_o1o2o3":   lambda: build_erlm(bkw, o1=True, o2=True, o3=True),
        }

        for method in args.methods:
            log.info(f"  → running {method}…")
            r = run_sample(
                model_fn=method_fns[method],
                document=doc, question=question, gold=gold,
                sample_id=sample_id, method=method,
                vllm_url=args.vllm_url, quantization=quant,
                vllm_client=vllm_client,
            )
            results.append(r)
            status = "✓" if not r.error else "✗"
            log.info(f"  {status} {method}: EM={r.exact_match:.2f} F1={r.f1:.2f} "
                     f"tok={r.tokens_used} ttft={r.ttft_ms:.0f}ms "
                     f"kv_hit={r.kv_cache_hit_rate:.1f}% "
                     f"speedup={r.speedup_vs_ollama:.2f}x {r.wall_clock_s:.1f}s")
            if r.error:
                log.warning(f"  ✗ {method}: {r.error}")

    _print_summary(results, args.methods, quant)

    # Save results
    os.makedirs(args.out_dir, exist_ok=True)
    jsonl_path = os.path.join(args.out_dir, f"{label}_{ts}.jsonl")
    with open(jsonl_path, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
    log.info(f"\nResults saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
