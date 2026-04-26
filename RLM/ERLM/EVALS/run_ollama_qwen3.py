"""
run_ollama_qwen3.py — LongBench-v2 CodeQA eval using Qwen3-8B via Ollama
=========================================================================
Evaluates four methods on LongBench-v2 CodeQA samples using Qwen3-8B
running locally via Ollama (http://localhost:11434).

Qwen3-8B context window: 128K tokens ≈ 480K chars (at ~3.75 chars/token).
All selected docs (800K–4M chars) exceed the model's context, so base_model
always truncates and RLM/ERLM iterative retrieval is meaningful for every sample.

Methods
-------
  base_model     — Direct Ollama call, doc truncated to 450K chars
  rlm_baseline   — RLM iterative REPL loop
  erlm_o1o2      — ERLM with O1 (TF-IDF indexer) + O2 (budget controller)
  erlm_o1o2o3    — ERLM with O1 + O2 + O3 (async parallel subcalls)

Requirements
------------
  ollama serve                          # Ollama must be running
  ollama pull qwen3:8b                  # model must be pulled

Usage
-----
    python run_ollama_qwen3.py --n 50
    python run_ollama_qwen3.py --n 3 --methods base_model rlm_baseline   # quick test
    python run_ollama_qwen3.py --n 50 --model_name qwen3:14b             # larger variant
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
# Model constants — Qwen3-8B via Ollama
# ---------------------------------------------------------------------------

_MODEL_NAME       = "qwen3:8b"
_MODEL_LABEL      = "ollama_qwen3_8b"

# Ollama endpoint
_OLLAMA_BASE_URL  = "http://localhost:11434/v1"

# Context: 128K tokens; at ~3.75 chars/token → ~480K chars; truncate safely at 450K
_BASE_MODEL_DOC_CHARS = 450_000

# RLM loop settings — local Ollama can be slow, give generous timeout
_MAX_ITERATIONS = 3
_MAX_TIMEOUT_S  = 600.0   # 10 min per sample; local inference varies widely

# CodeQA sample selection — same 50 docs across all models (seed=42)
# All docs exceed Qwen3-8B context → every sample puts pressure on RLM/ERLM
_MIN_DOC_LENGTH = 800_000
_MAX_DOC_LENGTH = 4_000_000

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(log_dir: str, ts: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{_MODEL_LABEL}_{ts}.log")
    logger = logging.getLogger(_MODEL_LABEL)
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
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SampleResult:
    model:               str
    dataset:             str
    sample_id:           str
    method:              str
    prediction:          str
    ground_truth:        str
    exact_match:         float
    f1:                  float
    tokens_used:         int
    input_tokens:        int
    output_tokens:       int
    tokens_per_sec:      float
    iterations:          int
    wall_clock_s:        float
    error:               str   = ""
    o1_chunks:           int   = 0
    early_terminated:    bool  = False
    termination_reason:  str   = ""
    o3_speedup_ratio:    float = 0.0
    o3_parallel_batches: int   = 0
    o3_calls_parallelized: int = 0

# ---------------------------------------------------------------------------
# Metrics
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
    # "the answer is C" / "answer: C" / "Answer (C)" / "(C)"
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
    # Standalone letter at end of text: "...therefore C." or "...is C\n"
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
# Client factory
# ---------------------------------------------------------------------------

def _make_backend_kwargs(model_name: str) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "base_url":   _OLLAMA_BASE_URL,
        "api_key":    "ollama",   # Ollama accepts any non-empty string
    }

# ---------------------------------------------------------------------------
# Base-model wrapper
# ---------------------------------------------------------------------------

class _BaseModelWrapper:
    def __init__(self, backend_kwargs: dict[str, Any]) -> None:
        from rlm.clients import get_client
        self.client = get_client("ollama", backend_kwargs)
        self._model_name = backend_kwargs["model_name"]

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
        response_text = self.client.completion(prompt)
        elapsed = time.perf_counter() - t0
        usage   = self.client.get_usage_summary() if hasattr(self.client, "get_usage_summary") else None
        return RLMChatCompletion(
            root_model=self._model_name, prompt=prompt, response=response_text,
            usage_summary=usage or UsageSummary(model_usage={}),
            execution_time=elapsed,
        )

# ---------------------------------------------------------------------------
# RLM / ERLM builders
# ---------------------------------------------------------------------------

def build_base_model(bkw: dict[str, Any]) -> _BaseModelWrapper:
    return _BaseModelWrapper(bkw)

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
        max_tokens=50_000,          # O2 budget: ~10% of 128K Qwen3 context (450K chars cap)
        enable_indexing=o1, enable_budget=o2, enable_async=o3,
        enable_kv_cache=False, enable_fp8=False,
        indexer_chunk_size=1000,    # Fix O1: was 2000; 1000 chars × 3 = 3K chars/query
        indexer_overlap=100,        # proportional to chunk_size
        indexer_top_k=3,            # Fix O1: was 5; reduces per-query token cost 3.3×
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
) -> SampleResult:
    t0 = time.perf_counter()
    prediction = ""; tokens_used = 0; input_tokens = 0; output_tokens = 0
    tokens_per_sec = 0.0; iterations = 0; error = ""; model = None
    o1_chunks = 0; early_terminated = False; termination_reason = ""
    o3_speedup_ratio = 0.0; o3_parallel_batches = 0; o3_calls_parallelized = 0

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

        # Iteration count: prefer on_iteration_complete counter over metadata
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
            o3_speedup_ratio      = stats.get("speedup_ratio", 0.0)
            o3_parallel_batches   = stats.get("total_parallel_batches", 0)
            o3_calls_parallelized = stats.get("total_calls_parallelized", 0)

        log.info(f"  [pred] gold={gold!r}  pred={prediction[:200]!r}")
        log.info(
            f"  [tok]  total={tokens_used}  in={input_tokens}  out={output_tokens}  "
            f"chunks={o1_chunks}  early_stop={early_terminated}  o3_batches={o3_parallel_batches}"
        )

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        log.error(f"[{method}] sample={sample_id} FAILED: {error}")
        log.debug(traceback.format_exc())

    wall = time.perf_counter() - t0
    em   = exact_match(prediction, gold)
    f1   = f1_score(prediction, gold)
    tokens_per_sec = round(tokens_used / wall, 1) if wall > 0 else 0.0

    return SampleResult(
        model=_MODEL_NAME, dataset="codeqa", sample_id=sample_id, method=method,
        prediction=prediction[:500], ground_truth=gold,
        exact_match=em, f1=f1,
        tokens_used=tokens_used, input_tokens=input_tokens, output_tokens=output_tokens,
        tokens_per_sec=tokens_per_sec, iterations=iterations,
        wall_clock_s=round(wall, 2), error=error,
        o1_chunks=o1_chunks, early_terminated=early_terminated,
        termination_reason=termination_reason, o3_speedup_ratio=o3_speedup_ratio,
        o3_parallel_batches=o3_parallel_batches, o3_calls_parallelized=o3_calls_parallelized,
    )

# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def _print_summary(results: list[SampleResult], methods: list[str]) -> None:
    from collections import defaultdict
    import statistics

    log.info("\n" + "="*120)
    log.info(f"SUMMARY — {_MODEL_NAME} | CodeQA / LongBench-v2")
    log.info("="*120)

    by_method: dict[str, list[SampleResult]] = defaultdict(list)
    for r in results:
        by_method[r.method].append(r)

    log.info("\n── Table 1: Quality & Token Efficiency ──")
    h1 = (f"{'Method':<20} {'N':>3} {'OK':>3} {'EM':>6} {'F1':>6} "
          f"{'TotalTok':>9} {'InTok':>7} {'OutTok':>7} {'Tok/s':>7} {'Time(s)':>8}")
    log.info(h1); log.info("-" * len(h1))

    baseline_tok = None; baseline_time = None
    for method in methods:
        rows = by_method.get(method, [])
        if not rows: continue
        ok = [r for r in rows if not r.error]
        if not ok:
            log.info(f"{method:<20} {len(rows):>3} {'0':>3}  — all errored"); continue
        em  = statistics.mean(r.exact_match for r in ok)
        f1  = statistics.mean(r.f1 for r in ok)
        tok = statistics.mean(r.tokens_used for r in ok)
        inp = statistics.mean(r.input_tokens for r in ok)
        out = statistics.mean(r.output_tokens for r in ok)
        tps = statistics.mean(r.tokens_per_sec for r in ok)
        wt  = statistics.mean(r.wall_clock_s for r in rows)
        if method == "rlm_baseline": baseline_tok = tok; baseline_time = wt
        log.info(f"{method:<20} {len(rows):>3} {len(ok):>3} {em:>6.3f} {f1:>6.3f} "
                 f"{tok:>9.0f} {inp:>7.0f} {out:>7.0f} {tps:>7.1f} {wt:>8.1f}")
    log.info("-" * len(h1))

    log.info("\n── Table 2: Optimization Verification ──")
    h2 = (f"{'Method':<20} {'TokRedux%':>10} {'SpeedupVsBase':>14} "
          f"{'TokEff(EM/1Ktok)':>17} {'O1Chunks':>9} {'O2FireRate':>11} {'O3Utilized':>11}")
    log.info(h2); log.info("-" * len(h2))
    for method in methods:
        rows = by_method.get(method, [])
        if not rows: continue
        ok = [r for r in rows if not r.error]
        if not ok: continue
        tok = statistics.mean(r.tokens_used for r in ok)
        wt  = statistics.mean(r.wall_clock_s for r in rows)
        em  = statistics.mean(r.exact_match for r in ok)
        tok_redux  = ((baseline_tok - tok) / baseline_tok * 100) if baseline_tok else 0.0
        wt_speedup = (baseline_time / wt) if (baseline_time and wt > 0) else 1.0
        tok_eff    = (em / (tok / 1000)) if tok > 0 else 0.0
        o1c  = statistics.mean(r.o1_chunks for r in ok)
        o2fr = statistics.mean(1.0 if r.early_terminated else 0.0 for r in ok) * 100
        o3ut = statistics.mean(1.0 if r.o3_parallel_batches > 0 else 0.0 for r in ok) * 100
        log.info(f"{method:<20} {tok_redux:>+9.1f}%  {wt_speedup:>13.2f}x "
                 f"{tok_eff:>17.4f}  {o1c:>9.0f}  {o2fr:>10.1f}%  {o3ut:>10.1f}%")
    log.info("-" * len(h2))

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def _save(results: list[SampleResult], out_dir: str, ts: str) -> None:
    path = os.path.join(out_dir, f"{_MODEL_LABEL}_{ts}.jsonl")
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_ALL_METHODS     = ["base_model", "rlm_baseline", "erlm_o1o2", "erlm_o1o2o3"]
_DEFAULT_METHODS = ["base_model", "rlm_baseline", "erlm_o1o2", "erlm_o1o2o3"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"CodeQA eval — {_MODEL_NAME} via Ollama"
    )
    parser.add_argument("--n",          type=int, default=50,
                        help="Number of CodeQA samples (default: 50)")
    parser.add_argument("--methods",    nargs="+", default=_DEFAULT_METHODS,
                        choices=_ALL_METHODS)
    parser.add_argument("--model_name", type=str, default=_MODEL_NAME,
                        help="Ollama model tag (default: qwen3:8b)")
    parser.add_argument("--out",        default=f"results/{_MODEL_LABEL}",
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    global log
    log = _setup_logging(args.out, ts)

    log.info("=" * 70)
    log.info(f"Model          : {args.model_name}")
    log.info(f"Backend        : Ollama ({_OLLAMA_BASE_URL})")
    log.info(f"Context window : 128K tokens (~480K chars)")
    log.info(f"Base-model cap : {_BASE_MODEL_DOC_CHARS:,} chars")
    log.info(f"Samples        : {args.n}")
    log.info(f"Methods        : {args.methods}")
    log.info(f"Doc length     : {_MIN_DOC_LENGTH:,} – {_MAX_DOC_LENGTH:,} chars")
    log.info(f"Max iterations : {_MAX_ITERATIONS}")
    log.info(f"Timeout/sample : {_MAX_TIMEOUT_S}s")
    log.info(f"Output dir     : {args.out}")
    log.info("=" * 70)

    bkw = _make_backend_kwargs(args.model_name)

    def _factory(method: str):
        if method == "base_model":    return lambda: build_base_model(bkw)
        if method == "rlm_baseline":  return lambda: build_rlm_baseline(bkw)
        if method == "erlm_o1o2":     return lambda: build_erlm(bkw, o1=True, o2=True, o3=False)
        if method == "erlm_o1o2o3":   return lambda: build_erlm(bkw, o1=True, o2=True, o3=True)
        raise ValueError(method)

    import warnings; warnings.filterwarnings("ignore")
    from benchmarks.longbench_codeqa import CodeQADataset

    ds = CodeQADataset(
        max_samples=args.n,
        min_doc_length=_MIN_DOC_LENGTH,
        max_doc_length=_MAX_DOC_LENGTH,
    ).load()

    if not ds:
        log.error(
            f"No samples found with doc length {_MIN_DOC_LENGTH:,}–{_MAX_DOC_LENGTH:,} chars."
        )
        return

    log.info(f"Loaded {len(ds)} samples. Running {len(args.methods)} methods…")

    all_results: list[SampleResult] = []

    for i, sample in enumerate(ds):
        log.info(f"[codeqa] Sample {i+1}/{len(ds)} id={sample.id}  "
                 f"doc_chars={len(sample.document):,}  q={sample.question[:80]!r}")

        for method in args.methods:
            log.info(f"  → running {method}…")
            result = run_sample(
                model_fn=_factory(method),
                document=sample.document,
                question=sample.question,
                gold=sample.answer,
                sample_id=str(sample.id),
                method=method,
            )
            all_results.append(result)
            if result.error:
                log.warning(f"  ✗ {method}: {result.error[:120]}")
            else:
                log.info(f"  ✓ {method}: EM={result.exact_match:.2f} F1={result.f1:.2f} "
                         f"tok={result.tokens_used} iters={result.iterations} "
                         f"{result.wall_clock_s:.1f}s")

        _save(all_results, args.out, ts)

    _print_summary(all_results, args.methods)
    log.info(f"\nResults saved to {args.out}/")


if __name__ == "__main__":
    main()
