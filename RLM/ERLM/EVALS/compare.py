"""
compare.py — LongBench-v2 CodeQA evaluation: base model vs RLM vs ERLM
=======================================================================
Evaluates six methods on LongBench-v2 CodeQA samples (long-context QA,
docs typically 500K–5M characters, answers are single A/B/C/D letters).

Methods
-------
  base_model     — Direct LLM call, no RLM loop (document truncated to ctx window)
  rlm_baseline   — RLM loop, base model
  erlm_o1o2      — ERLM with O1 (TF-IDF indexer) + O2 (budget controller), base model
  erlm_o1o2o3    — ERLM with O1 + O2 + O3 (async batching), base model
  rlm_finetuned  — RLM loop, fine-tuned RLM model
  erlm_finetuned — ERLM with O1 + O2, fine-tuned RLM model

Backends
--------
  ollama  — Local Ollama server (default: qwen3:8b / rlm-qwen3-8b-v0.1-gguf)
  vertex  — Vertex AI MaaS (requires GCP auth)

Usage
-----
    # Core 4 methods on Ollama:
    python compare.py --backend ollama --n 5

    # All 6 methods including fine-tuned:
    python compare.py --backend ollama --n 5 --methods base_model rlm_baseline erlm_o1o2 erlm_o1o2o3 rlm_finetuned erlm_finetuned

    # Vertex AI:
    gcloud auth application-default login
    python compare.py --backend vertex --project YOUR_PROJECT_ID --n 5
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
# Logging — writes to stdout AND a timestamped log file
# ---------------------------------------------------------------------------

def _setup_logging(log_dir: str = "results") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"compare_{ts}.log")

    logger = logging.getLogger("compare")
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
    logger.info(f"Log file: {log_path}")
    return logger


log = logging.getLogger("compare")   # populated after _setup_logging()


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ERLM_DIR = os.path.join(_THIS_DIR, "..")
_BASELINE_DIR = os.path.join(_ERLM_DIR, "..", "BASELINE")

sys.path.insert(0, _BASELINE_DIR)
sys.path.insert(0, _ERLM_DIR)
sys.path.insert(0, _THIS_DIR)

# ---------------------------------------------------------------------------
# Vertex AI token management
# ---------------------------------------------------------------------------

_TOKEN_CACHE: dict[str, Any] = {"token": None, "fetched_at": 0.0}
_TOKEN_TTL = 2700  # refresh every 45 min (tokens last 60 min)

VERTEX_MODEL = "qwen/qwen3-coder-480b-a35b-instruct-maas"
VERTEX_REGION = os.environ.get("VERTEX_REGION", "us-south1")


def get_vertex_token() -> str:
    """Return a cached or fresh Vertex AI access token."""
    now = time.time()
    if _TOKEN_CACHE["token"] is None or (now - _TOKEN_CACHE["fetched_at"]) > _TOKEN_TTL:
        try:
            import google.auth
            import google.auth.transport.requests
        except ImportError:
            raise ImportError("pip install google-auth")
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(google.auth.transport.requests.Request())
        _TOKEN_CACHE["token"] = credentials.token
        _TOKEN_CACHE["fetched_at"] = now
        print(f"[token] Refreshed Vertex AI token at {datetime.now():%H:%M:%S}", flush=True)
    return _TOKEN_CACHE["token"]


def vertex_backend_kwargs(project_id: str) -> dict[str, Any]:
    """Return backend_kwargs for RLM/ERLM using Vertex AI OpenAI-compatible endpoint."""
    token = get_vertex_token()
    base_url = (
        f"https://{VERTEX_REGION}-aiplatform.googleapis.com/v1/projects/"
        f"{project_id}/locations/{VERTEX_REGION}/endpoints/openapi"
    )
    return {
        "api_key": token,
        "base_url": base_url,
        "model_name": VERTEX_MODEL,
    }


def ollama_backend_kwargs(model: str = "qwen3:8b") -> dict[str, Any]:
    """Return backend_kwargs for RLM/ERLM using a local Ollama server.

    Uses the 'ollama' backend type so get_client returns OllamaClient,
    which injects think:false into every request to suppress chain-of-thought.
    """
    return {
        "model_name": model,
        "base_url": "http://localhost:11434/v1",
    }


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SampleResult:
    dataset: str
    sample_id: str
    method: str
    prediction: str
    ground_truth: str
    exact_match: float
    f1: float
    tokens_used: int        # total = input + output
    input_tokens: int       # prompt tokens (what O1 reduces)
    output_tokens: int      # generated tokens (what O2 reduces)
    tokens_per_sec: float   # throughput: tokens_used / wall_clock_s
    iterations: int
    wall_clock_s: float
    error: str = ""
    # O1 — TF-IDF indexer
    o1_chunks: int = 0              # number of TF-IDF chunks built for this doc
    # O2 — Budget controller
    early_terminated: bool = False  # did O2 fire early termination?
    termination_reason: str = ""    # "critical_budget" | "low_budget" | "low_productivity" | ""
    # O3 — Async subcall manager
    o3_speedup_ratio: float = 0.0   # sequential_equivalent / actual_parallel
    o3_parallel_batches: int = 0    # number of llm_query_batched calls measured
    o3_calls_parallelized: int = 0  # total individual prompts that ran in parallel


# ---------------------------------------------------------------------------
# Metrics (inline to avoid import issues)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    import re, string
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _extract_mc_letter(text: str) -> str:
    """Extract the multiple-choice answer letter (A/B/C/D) from model output.

    Handles formats like:
      "D"  /  "The answer is D"  /  "D. ..."  /  "(D)"  /  "answer: D"
    Returns the letter lowercase, or the full normalized text if no letter found.
    """
    import re
    # Look for an isolated A/B/C/D (case-insensitive) — prefer the first match
    # preceded by start, space, colon, parens, or period
    m = re.search(r'(?:^|[\s:()\[\].,"\'`])\b([A-Da-d])\b(?:[\s:).,"\'`]|$)', text)
    if m:
        return m.group(1).lower()
    return _normalize(text)


def exact_match(pred: str, gold: str) -> float:
    gold_n = _normalize(gold)
    # If gold is a single letter it's multiple-choice — extract letter from pred
    if len(gold_n) == 1 and gold_n in "abcd":
        return 1.0 if _extract_mc_letter(pred) == gold_n else 0.0
    return 1.0 if _normalize(pred) == gold_n else 0.0


def f1_score(pred: str, gold: str) -> float:
    from collections import Counter
    gold_n = _normalize(gold)
    # Multiple-choice: treat as exact match (F1 = EM for single-token answers)
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
    recall = common / len(g_toks)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# RLM / ERLM builders — same model for main + subcalls (no other_backends)
# ---------------------------------------------------------------------------

_MAX_ITERATIONS = 2      # LLM rounds per sample — CPU Qwen3-8B is ~300s/iter; 2 × 300s ≈ 600s
_MAX_TIMEOUT_S  = 1200.0 # 20-min hard wall-clock cap — enough for 2 iterations + TF-IDF index build

# Characters that fit in ~38K tokens (leaving headroom for system prompt + question)
# Qwen3-8B context = 40960 tokens; ~4 chars/token → ~152K chars safe limit
_BASE_MODEL_DOC_CHARS = 140_000


class _BaseModelWrapper:
    """Thin wrapper that calls the LLM directly (no RLM loop).

    Truncates the document to fit in the model's context window and asks
    the question directly.  This is the 'base_model' baseline — it shows
    what the raw LM can do without any RLM scaffolding.
    """

    def __init__(self, backend_kwargs: dict[str, Any], backend: str) -> None:
        from rlm.clients import get_client
        self.client = get_client(backend, backend_kwargs)
        self._usage_summary = None

    def completion(self, document: str, question: str) -> Any:
        from rlm.core.types import RLMChatCompletion, UsageSummary
        import time

        truncated = document[:_BASE_MODEL_DOC_CHARS]
        trunc_note = (
            f"\n\n[Document truncated to {_BASE_MODEL_DOC_CHARS:,} chars "
            f"of {len(document):,} total due to context window limit.]"
            if len(document) > _BASE_MODEL_DOC_CHARS else ""
        )
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the question below based solely "
                    "on the provided document. For multiple-choice questions, respond with "
                    "only the letter (A, B, C, or D)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Document:\n{truncated}{trunc_note}\n\n"
                    f"Question: {question}\n\n"
                    "Answer (letter only):"
                ),
            },
        ]
        t0 = time.perf_counter()
        response_text = self.client.completion(prompt)
        elapsed = time.perf_counter() - t0

        usage = self.client.get_usage_summary() if hasattr(self.client, "get_usage_summary") else None

        return RLMChatCompletion(
            root_model=getattr(self.client, "model_name", "unknown"),
            prompt=prompt,
            response=response_text,
            usage_summary=usage or UsageSummary(model_usage={}),
            execution_time=elapsed,
        )


def build_base_model(backend_kwargs: dict[str, Any], backend: str = "ollama") -> _BaseModelWrapper:
    """Direct LLM call with no RLM loop."""
    return _BaseModelWrapper(backend_kwargs, backend)


def build_rlm_baseline(backend_kwargs: dict[str, Any], backend: str = "openai") -> Any:
    """Plain RLM with no ERLM optimisations."""
    from rlm.core.rlm import RLM
    return RLM(
        backend=backend,
        backend_kwargs=backend_kwargs,
        max_depth=1,
        max_iterations=_MAX_ITERATIONS,
        max_timeout=_MAX_TIMEOUT_S,
    )


def build_erlm(backend_kwargs: dict[str, Any], o1: bool, o2: bool, o3: bool,
               backend: str = "openai") -> Any:
    """EnhancedRLM with requested optimisation flags."""
    from erlm import EnhancedRLM
    return EnhancedRLM(
        backend=backend,
        backend_kwargs=backend_kwargs,
        max_depth=1,
        max_iterations=_MAX_ITERATIONS,
        max_timeout=_MAX_TIMEOUT_S,
        enable_indexing=o1,
        enable_budget=o2,
        enable_async=o3,
        enable_kv_cache=False,
        enable_fp8=False,
    )


# ---------------------------------------------------------------------------
# Rate-limit retry logic
# ---------------------------------------------------------------------------

# Vertex AI / OpenAI error types that warrant a retry with backoff
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
# Initial wait for 429; doubles each attempt.
# Vertex AI MaaS Qwen3-480B has strict per-minute quotas on new projects;
# start at 70s (>1 min) so the first retry clears the per-minute window.
_RATE_LIMIT_BASE_WAIT = 70   # seconds
_MAX_RETRIES = 8


def _is_rate_limit(exc: Exception) -> bool:
    """Return True if *exc* is a 429 / rate-limit / quota error."""
    msg = str(exc).lower()
    if any(k in msg for k in ("429", "rate limit", "quota", "resource exhausted", "too many requests")):
        return True
    # openai SDK raises RateLimitError
    typ = type(exc).__name__
    if typ in ("RateLimitError", "APIStatusError", "APIConnectionError"):
        status = getattr(exc, "status_code", None)
        if status in _RETRYABLE_STATUS:
            return True
    return False


def _is_retryable(exc: Exception) -> bool:
    """Return True for transient server errors (5xx) in addition to rate limits."""
    if _is_rate_limit(exc):
        return True
    msg = str(exc).lower()
    typ = type(exc).__name__
    if any(k in msg for k in ("502", "503", "504", "service unavailable", "bad gateway")):
        return True
    if typ in ("APIConnectionError", "APITimeoutError", "InternalServerError"):
        return True
    return False


def _run_with_retry(model_fn, document: str, question: str) -> Any:
    """
    Call model.completion() with exponential backoff on rate-limit / transient errors.

    Retries up to _MAX_RETRIES times. On 429 waits _RATE_LIMIT_BASE_WAIT * 2^attempt
    seconds before retrying. Re-instantiates the model on each retry (fresh token).
    """
    wait = _RATE_LIMIT_BASE_WAIT
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            model = model_fn()
            # _BaseModelWrapper takes (document, question); RLM takes (document, root_prompt=question)
            if isinstance(model, _BaseModelWrapper):
                return model, model.completion(document, question)
            return model, model.completion(document, root_prompt=question)
        except Exception as exc:
            last_exc = exc
            if attempt == _MAX_RETRIES:
                break
            if _is_rate_limit(exc):
                log.warning(
                    f"Rate limit hit (attempt {attempt+1}/{_MAX_RETRIES}). "
                    f"Waiting {wait}s before retry… ({exc})"
                )
                time.sleep(wait)
                wait = min(wait * 2, 600)   # cap at 10 minutes
                # Also refresh token in case it expired during wait
                get_vertex_token()
            elif _is_retryable(exc):
                log.warning(
                    f"Transient error (attempt {attempt+1}/{_MAX_RETRIES}). "
                    f"Waiting {wait}s … ({exc})"
                )
                time.sleep(wait)
                wait = min(wait * 2, 300)
            else:
                # Non-retryable (e.g. bad prompt, auth failure) — fail immediately
                raise

    raise last_exc  # re-raise after exhausting retries


# ---------------------------------------------------------------------------
# Run one sample with one method
# ---------------------------------------------------------------------------

def run_sample(
    model_fn,          # callable() → fresh RLM/ERLM instance
    document: str,
    question: str,
    gold: str,
    dataset: str,
    sample_id: str,
    method: str,
) -> SampleResult:
    """Run one sample with retry on rate limits. Never crashes the outer loop."""
    t0 = time.perf_counter()
    prediction = ""
    tokens_used = 0
    input_tokens = 0
    output_tokens = 0
    tokens_per_sec = 0.0
    iterations = 0
    error = ""
    model = None

    # Optimization-specific fields
    o1_chunks = 0
    early_terminated = False
    termination_reason = ""
    o3_speedup_ratio = 0.0
    o3_parallel_batches = 0
    o3_calls_parallelized = 0

    try:
        model, result = _run_with_retry(model_fn, document, question)
        # RLMChatCompletion uses .response for the final answer text
        prediction = (
            getattr(result, "response", None)
            or getattr(result, "final_answer", None)
            or ""
        )

        # Extract input/output tokens separately from usage_summary
        us = getattr(result, "usage_summary", None) or getattr(result, "usage", None)
        if us is not None:
            input_tokens  = getattr(us, "total_input_tokens", 0)
            output_tokens = getattr(us, "total_output_tokens", 0)
            tokens_used   = input_tokens + output_tokens
        # Iteration count: embedded in metadata when a logger is attached; otherwise estimate from execution_time
        if hasattr(result, "metadata") and isinstance(result.metadata, dict):
            iters_list = result.metadata.get("iterations", [])
            iterations = len(iters_list) if isinstance(iters_list, list) else 0

        # --- O1: TF-IDF chunk count ---
        if hasattr(model, "_indexer") and model._indexer is not None:
            o1_chunks = len(model._indexer.chunk_offsets)

        # --- O2: early termination ---
        if hasattr(model, "_budget_controller") and model._budget_controller is not None:
            reason = model._budget_controller.termination_reason or ""
            termination_reason = reason
            early_terminated = bool(reason)

        # --- O3: async speedup stats ---
        if hasattr(model, "_async_manager") and model._async_manager is not None:
            stats = model._async_manager.get_speedup_stats()
            o3_speedup_ratio = stats.get("speedup_ratio", 0.0)
            o3_parallel_batches = stats.get("total_parallel_batches", 0)
            o3_calls_parallelized = stats.get("total_calls_parallelized", 0)

        log.debug(
            f"[{method}] sample={sample_id} em={exact_match(prediction, gold):.2f} "
            f"tok={tokens_used} iters={iterations} chunks={o1_chunks} "
            f"early_stop={early_terminated}({termination_reason}) "
            f"speedup={o3_speedup_ratio:.2f}x batches={o3_parallel_batches}"
        )
        log.info(
            f"  [pred] gold={gold!r}  pred={prediction[:200]!r}"
        )
        log.info(
            f"  [tok]  total={tokens_used}  in={input_tokens}  out={output_tokens}  "
            f"chunks={o1_chunks}  early_stop={early_terminated}  o3_batches={o3_parallel_batches}"
        )

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        log.error(f"[{method}] sample={sample_id} FAILED after retries: {error}")
        log.debug(traceback.format_exc())

    wall = time.perf_counter() - t0
    em = exact_match(prediction, gold)
    f1 = f1_score(prediction, gold)
    tokens_per_sec = round(tokens_used / wall, 1) if wall > 0 else 0.0

    return SampleResult(
        dataset=dataset,
        sample_id=sample_id,
        method=method,
        prediction=prediction[:500],
        ground_truth=gold,
        exact_match=em,
        f1=f1,
        tokens_used=tokens_used,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tokens_per_sec=tokens_per_sec,
        iterations=iterations,
        wall_clock_s=round(wall, 2),
        error=error,
        o1_chunks=o1_chunks,
        early_terminated=early_terminated,
        termination_reason=termination_reason,
        o3_speedup_ratio=o3_speedup_ratio,
        o3_parallel_batches=o3_parallel_batches,
        o3_calls_parallelized=o3_calls_parallelized,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_ALL_METHODS = ["base_model", "rlm_baseline", "erlm_o1o2", "erlm_o1o2o3",
                "rlm_finetuned", "erlm_finetuned"]
_DEFAULT_FINETUNED_MODEL = "hf.co/cameronbergh/rlm-qwen3-8b-v0.1-gguf:Q8_0"


def main():
    parser = argparse.ArgumentParser(description="CodeQA eval: base model vs RLM vs ERLM")
    parser.add_argument("--backend", default="ollama", choices=["ollama", "vertex"],
                        help="LLM backend: 'ollama' (local) or 'vertex' (GCP). Default: ollama")
    parser.add_argument("--ollama-model", default="qwen3:8b",
                        help="Ollama base model (default: qwen3:8b)")
    parser.add_argument("--finetuned-model", default=_DEFAULT_FINETUNED_MODEL,
                        help="Ollama fine-tuned model for rlm_finetuned / erlm_finetuned methods")
    parser.add_argument("--n", type=int, default=5,
                        help="Samples from CodeQA (default: 5)")
    parser.add_argument("--methods", nargs="+",
                        default=["base_model", "rlm_baseline", "erlm_o1o2", "erlm_o1o2o3"],
                        choices=_ALL_METHODS)
    parser.add_argument("--project", default=os.environ.get("VERTEX_PROJECT_ID"),
                        help="GCP project ID — required for --backend vertex")
    parser.add_argument("--out", default="results", help="Output directory")
    args = parser.parse_args()

    if args.backend == "vertex" and not args.project:
        parser.error(
            "--backend vertex requires --project YOUR_PROJECT or export VERTEX_PROJECT_ID=..."
        )

    os.makedirs(args.out, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set up logging (stdout + file)
    global log
    log = _setup_logging(args.out)

    if args.backend == "ollama":
        bkw = ollama_backend_kwargs(args.ollama_model)
        bkw_ft = ollama_backend_kwargs(args.finetuned_model)
        backend_type = "ollama"
        log.info(f"Backend        : Ollama (local)")
        log.info(f"Base model     : {args.ollama_model}")
        log.info(f"Finetuned model: {args.finetuned_model}")
        log.info(f"Endpoint       : http://localhost:11434/v1")
    else:
        bkw = vertex_backend_kwargs(args.project)
        bkw_ft = bkw   # Vertex uses same model for all methods
        backend_type = "openai"
        log.info(f"Backend        : Vertex AI")
        log.info(f"Model          : {VERTEX_MODEL}")
        log.info(f"GCP project    : {args.project}")
        log.info("Checking Vertex AI credentials…")
        get_vertex_token()
        log.info("Token OK.")

    log.info(f"Dataset        : CodeQA (LongBench-v2)")
    log.info(f"Samples        : {args.n}")
    log.info(f"Methods        : {args.methods}")
    log.info(f"Output dir     : {args.out}")

    # Method dispatch — returns a zero-arg callable that builds a fresh model instance
    def _factory(method: str):
        if method == "base_model":
            return lambda: build_base_model(bkw, backend=backend_type)
        if method == "rlm_baseline":
            return lambda: build_rlm_baseline(bkw, backend=backend_type)
        if method == "erlm_o1o2":
            return lambda: build_erlm(bkw, o1=True, o2=True, o3=False, backend=backend_type)
        if method == "erlm_o1o2o3":
            return lambda: build_erlm(bkw, o1=True, o2=True, o3=True, backend=backend_type)
        if method == "rlm_finetuned":
            return lambda: build_rlm_baseline(bkw_ft, backend=backend_type)
        if method == "erlm_finetuned":
            return lambda: build_erlm(bkw_ft, o1=True, o2=True, o3=False, backend=backend_type)
        raise ValueError(method)

    all_results: list[SampleResult] = []

    log.info(f"{'='*60}")
    log.info(f"Dataset: CODEQA  (n={args.n}, docs > 512K chars)")
    log.info(f"{'='*60}")

    import warnings
    warnings.filterwarnings("ignore")

    from benchmarks.longbench_codeqa import CodeQADataset
    # Cap docs at 2M chars: large enough to require RLM, small enough for CPU inference
    ds = CodeQADataset(max_samples=args.n, min_doc_length=512_000, max_doc_length=2_000_000).load()

    log.info(f"Loaded {len(ds)} samples. Running {len(args.methods)} methods…")

    for i, sample in enumerate(ds):
        doc = sample.document
        q   = sample.question
        ans = sample.answer
        sid = sample.id

        log.info(f"[codeqa] Sample {i+1}/{len(ds)} id={sid}  "
                 f"doc_chars={len(doc):,}  q={q[:80]!r}")

        for method in args.methods:
            log.info(f"  → running {method}…")
            result = run_sample(
                model_fn=_factory(method),
                document=doc,
                question=q,
                gold=ans,
                dataset="codeqa",
                sample_id=str(sid),
                method=method,
            )
            all_results.append(result)
            if result.error:
                log.warning(f"  ✗ {method}: {result.error[:120]}")
            else:
                log.info(f"  ✓ {method}: EM={result.exact_match:.2f} F1={result.f1:.2f} "
                         f"tok={result.tokens_used} iters={result.iterations} {result.wall_clock_s:.1f}s")

        # Save after every sample so a crash loses nothing
        _save(all_results, args.out, ts)
        log.debug(f"  [saved] {len(all_results)} results so far")

    # Final summary
    _print_summary(all_results, args.methods)
    log.info(f"Results saved to {args.out}/")


def _save(results: list[SampleResult], out_dir: str, ts: str):
    path = os.path.join(out_dir, f"compare_{ts}.jsonl")
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")


def _print_summary(results: list[SampleResult], methods: list[str]):
    from collections import defaultdict
    import statistics

    log.info("\n" + "="*120)
    log.info("SUMMARY — Quality & Efficiency (CodeQA / LongBench-v2)")
    log.info("="*120)

    by_method: dict[str, list[SampleResult]] = defaultdict(list)
    for r in results:
        by_method[r.method].append(r)

    # --- Table 1: Core quality + efficiency ---
    log.info("\n── Table 1: Quality & Token Efficiency ──")
    h1 = (f"{'Method':<20} {'N':>3} {'OK':>3} {'EM':>6} {'F1':>6} "
          f"{'TotalTok':>9} {'InTok':>7} {'OutTok':>7} {'Tok/s':>7} {'Time(s)':>8}")
    log.info(h1)
    log.info("-" * len(h1))

    baseline_tok = None
    baseline_time = None

    for method in methods:
        rows = by_method.get(method, [])
        if not rows:
            continue
        ok = [r for r in rows if not r.error]
        if not ok:
            log.info(f"{method:<20} {len(rows):>3} {'0':>3}  — all errored")
            continue

        em  = statistics.mean(r.exact_match for r in ok)
        f1  = statistics.mean(r.f1 for r in ok)
        tok = statistics.mean(r.tokens_used for r in ok)
        inp = statistics.mean(r.input_tokens for r in ok)
        out = statistics.mean(r.output_tokens for r in ok)
        tps = statistics.mean(r.tokens_per_sec for r in ok)
        wt  = statistics.mean(r.wall_clock_s for r in rows)

        if method == "rlm_baseline":
            baseline_tok  = tok
            baseline_time = wt

        log.info(f"{method:<20} {len(rows):>3} {len(ok):>3} {em:>6.3f} {f1:>6.3f} "
                 f"{tok:>9.0f} {inp:>7.0f} {out:>7.0f} {tps:>7.1f} {wt:>8.1f}")

    log.info("-" * len(h1))
    log.info("  InTok=avg input tokens (prompt size)  OutTok=avg output tokens (generation size)")
    log.info("  Tok/s=tokens per second throughput")

    # --- Table 2: Optimization verification ---
    log.info("\n── Table 2: Optimization Verification ──")
    h2 = (f"{'Method':<20} {'TokRedux%':>10} {'SpeedupVsBase':>14} "
          f"{'TokEff(EM/1Ktok)':>17} {'O1Chunks':>9} {'O2FireRate':>11} "
          f"{'O3Utilized':>11}")
    log.info(h2)
    log.info("-" * len(h2))

    for method in methods:
        rows = by_method.get(method, [])
        if not rows:
            continue
        ok = [r for r in rows if not r.error]
        if not ok:
            continue

        tok  = statistics.mean(r.tokens_used for r in ok)
        wt   = statistics.mean(r.wall_clock_s for r in rows)
        em   = statistics.mean(r.exact_match for r in ok)

        # Token reduction vs rlm_baseline
        tok_redux = ((baseline_tok - tok) / baseline_tok * 100) if baseline_tok else 0.0

        # Wall-clock speedup vs rlm_baseline
        wt_speedup = (baseline_time / wt) if (baseline_time and wt > 0) else 1.0

        # Token efficiency: correct answers per 1K tokens
        tok_eff = (em / (tok / 1000)) if tok > 0 else 0.0

        # O1: did indexer fire?
        chunks_vals = [r.o1_chunks for r in ok if r.o1_chunks > 0]
        avg_chunks = statistics.mean(chunks_vals) if chunks_vals else 0.0

        # O2: fire rate
        o2_rate = 100 * sum(r.early_terminated for r in ok) / len(ok)

        # O3: utilization — % samples where model actually called llm_query_batched
        o3_used = 100 * sum(1 for r in ok if r.o3_parallel_batches > 0) / len(ok)

        log.info(f"{method:<20} {tok_redux:>+10.1f}% {wt_speedup:>13.2f}x "
                 f"{tok_eff:>17.4f} {avg_chunks:>9.0f} {o2_rate:>10.1f}% "
                 f"{o3_used:>10.1f}%")

    log.info("-" * len(h2))
    log.info("  TokRedux%: token savings vs rlm_baseline (+ve = fewer tokens = better)")
    log.info("  SpeedupVsBase: wall-clock speedup vs rlm_baseline (>1.0x = faster)")
    log.info("  TokEff: EM score per 1K tokens (higher = more efficient use of compute)")
    log.info("  O1Chunks: avg TF-IDF chunks built (>0 confirms O1 active)")
    log.info("  O2FireRate: % samples where budget controller triggered early stop")
    log.info("  O3Utilized: % samples where model actually issued llm_query_batched calls")


if __name__ == "__main__":
    main()
