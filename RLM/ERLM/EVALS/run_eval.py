"""
run_eval.py — Main Evaluation Harness
======================================
Single entrypoint for comparing all methods (Vanilla, Compaction, ReAct,
RLM, and ERLM variants) on all three benchmarks: CodeQA, BrowseComp, Oolong.

Usage examples
--------------
    python run_eval.py --benchmark codeqa --method all --n_samples 10 --model gemini
    python run_eval.py --benchmark browsecomp --method rlm --n_samples 50 --model vllm
    python run_eval.py --benchmark oolong --method erlm_o1o2o3 --n_samples 20 --model ollama
    python run_eval.py --benchmark all --method vanilla --n_samples 5 --model openai --model_name gpt-4o
"""

from __future__ import annotations

import sys
import os
import argparse
import csv
import json
import re
import time
import traceback
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — must come before any local imports
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EVALS_DIR = _THIS_DIR
_ERLM_DIR = os.path.join(_THIS_DIR, "..")
_BASELINE_DIR = os.path.join(_ERLM_DIR, "..", "BASELINE")
_BENCHMARKS_DIR = os.path.join(_ERLM_DIR, "..", "BENCHMARKS")

sys.path.insert(0, _BASELINE_DIR)
sys.path.insert(0, _ERLM_DIR)
sys.path.insert(0, _EVALS_DIR)

# ---------------------------------------------------------------------------
# Optional tqdm
# ---------------------------------------------------------------------------

try:
    from tqdm import tqdm as _tqdm

    def _progress(iterable, desc: str = "", total: int = 0):
        return _tqdm(iterable, desc=desc, total=total or None)

except ImportError:
    class _FallbackProgress:
        def __init__(self, iterable, desc: str = "", total: int = 0):
            self._iterable = iterable
            self._desc = desc
            self._total = total
            self._idx = 0

        def __iter__(self):
            for item in self._iterable:
                self._idx += 1
                print(f"  [{self._desc}] {self._idx}/{self._total or '?'}", flush=True)
                yield item

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def _progress(iterable, desc: str = "", total: int = 0):
        return _FallbackProgress(iterable, desc=desc, total=total)

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------

from baselines import VanillaBaseline, CompactionBaseline, ReActBaseline, BaselineResult
from benchmarks import CodeQADataset, BrowseCompDataset, OolongDataset

# BASELINE RLM
from rlm.clients.gemini import GeminiClient
from rlm.clients.openai import OpenAIClient
from rlm.clients.base_lm import BaseLM
from rlm.core.rlm import RLM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_METHODS = [
    "vanilla",
    "compaction",
    "react",
    "rlm",
    "erlm_o1",
    "erlm_o2",
    "erlm_o3",
    "erlm_o1o2o3",
    "erlm_all",
]

ALL_BENCHMARKS = ["codeqa", "browsecomp", "oolong"]

# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------


def get_client(model: str, model_name: str | None = None) -> BaseLM:
    """Instantiate a :class:`BaseLM` for the given *model* shorthand.

    Parameters
    ----------
    model:
        One of ``"gemini"``, ``"ollama"``, ``"vllm"``, ``"openai"``.
    model_name:
        Optional override for the underlying model identifier passed to the
        provider API.

    Returns
    -------
    BaseLM
        A fully configured language model client.

    Raises
    ------
    ValueError
        When *model* is not a recognised shorthand.
    """
    if model == "gemini":
        return GeminiClient(model_name=model_name or "gemini-2.5-flash")

    if model == "ollama":
        effective_model = model_name or "llama3.2"
        return OpenAIClient(
            api_key="ollama",
            model_name=effective_model,
            base_url="http://localhost:11434/v1",
        )

    if model == "vllm":
        # Try importing the ERLM-optimised vLLM client; fall back to vanilla
        # OpenAIClient pointed at a local vLLM server if the optimised client
        # is not available.
        try:
            from optimisations.kv_prefix_cache import VLLMPrefixCachedClient  # type: ignore
            return VLLMPrefixCachedClient(model_name=model_name or "meta-llama/Llama-3.1-8B-Instruct")
        except ImportError:
            print(
                "[WARN] VLLMPrefixCachedClient not found in optimisations package; "
                "falling back to plain OpenAIClient pointed at http://localhost:8000/v1",
                flush=True,
            )
            return OpenAIClient(
                api_key="vllm",
                model_name=model_name or "meta-llama/Llama-3.1-8B-Instruct",
                base_url="http://localhost:8000/v1",
            )

    if model == "openai":
        return OpenAIClient(model_name=model_name or "gpt-4o-mini")

    if model == "vertex":
        # Qwen3-Coder-480B-A35B on Vertex AI — primary quality-eval model
        try:
            sys.path.insert(0, os.path.join(_ERLM_DIR, "clients"))
            from vertex_ai import VertexAIClient, QWEN3_CODER_480B  # type: ignore
            return VertexAIClient(
                model_name=model_name or QWEN3_CODER_480B,
            )
        except ImportError as e:
            raise ImportError(
                "VertexAIClient requires google-auth: pip install google-auth\n"
                f"Original error: {e}"
            )

    raise ValueError(
        f"Unknown model shorthand '{model}'. "
        "Choose from: gemini, ollama, vllm, openai, vertex"
    )


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------


def get_dataset(benchmark: str, n_samples: int | None = None) -> list[Any]:
    """Load samples for *benchmark*, filtered to docs that exceed the model context.

    Parameters
    ----------
    benchmark:
        One of ``"codeqa"``, ``"browsecomp"``, ``"oolong"``.
    n_samples:
        Hard cap on returned samples (None = all qualifying samples).
        For OOLONG this is per question_type (stratified).

    Returns
    -------
    list
        Dataset-specific sample dataclasses, all with doc length > 512K chars
        (= ~128K tokens, the Qwen3-Coder-480B-A35B context window).
    """
    if benchmark == "codeqa":
        # 119 qualifying samples (docs > 512K chars) across Code + Multi-Doc + Single-Doc QA
        ds = CodeQADataset(max_samples=n_samples, min_doc_length=512_000)
        return ds.load()

    if benchmark == "browsecomp":
        return BrowseCompDataset(max_samples=n_samples or 50).load()

    if benchmark == "oolong":
        # Stratified: 50 per question_type, only docs > 512K chars → 100 total
        per_type = n_samples or 50
        ds = OolongDataset(max_samples_per_type=per_type, min_doc_length=512_000)
        return ds.load()

    raise ValueError(
        f"Unknown benchmark '{benchmark}'. Choose from: {ALL_BENCHMARKS}"
    )


# ---------------------------------------------------------------------------
# Sample field accessors (normalise across different dataset schemas)
# ---------------------------------------------------------------------------


def _sample_id(sample: Any) -> str:
    return str(getattr(sample, "id", id(sample)))


def _sample_document(sample: Any) -> str:
    """Return the document/corpus field regardless of the attribute name."""
    for attr in ("document", "corpus", "context", "passage"):
        val = getattr(sample, attr, None)
        if val is not None:
            return str(val)
    return ""


def _sample_question(sample: Any) -> str:
    """Return the question/problem field."""
    for attr in ("question", "problem", "query", "input"):
        val = getattr(sample, attr, None)
        if val is not None:
            return str(val)
    return ""


def _sample_answer(sample: Any) -> str:
    """Return the ground-truth answer field."""
    for attr in ("answer", "gold_answer", "answers"):
        val = getattr(sample, attr, None)
        if val is not None:
            if isinstance(val, list):
                return str(val[0]) if val else ""
            return str(val)
    return ""


# ---------------------------------------------------------------------------
# ERLM: lazy import so the harness degrades gracefully when ERLM is partial
# ---------------------------------------------------------------------------


def _build_erlm(client: BaseLM, flags: dict[str, bool]) -> Any:
    """Build an EnhancedRLM (or wrapped RLM) respecting *flags*.

    Flags map optimisation IDs to booleans:
        o1 — PromptIndexer (TF-IDF search_context tool)
        o2 — AdaptiveBudgetController
        o3 — AsyncSubcallManager system-prompt addon
        o4 / o5 — reserved for future optimisations

    If the ERLM core module is not importable this function raises
    ``ImportError`` with a descriptive message.
    """
    # Try a dedicated EnhancedRLM wrapper first
    try:
        from erlm import EnhancedRLM  # type: ignore
        return EnhancedRLM(client=client, **flags)
    except ImportError:
        pass

    # Fall back to composing optimisations manually around the baseline RLM
    # -------------------------------------------------------------------------
    # Determine backend from client type
    backend = "gemini" if isinstance(client, GeminiClient) else "openai"
    backend_kwargs: dict[str, Any] = {"model_name": client.model_name}
    if backend == "openai":
        backend_kwargs["base_url"] = getattr(client, "base_url", None)

    custom_tools: dict[str, Any] = {}
    custom_system_prompt: str | None = None

    # O1: TF-IDF PromptIndexer — exposed as REPL tool (tool is index-agnostic;
    # the document is injected at run time via a wrapper)
    if flags.get("o1"):
        from optimisations.prompt_indexer import PromptIndexer  # type: ignore
        # We create the indexer at build time; it will be indexed per-call
        # by wrapping the RLM.completion call (see run_method).
        # Register a placeholder so we know O1 is active.
        custom_tools["_erlm_o1_enabled"] = True

    # O2: AdaptiveBudgetController — purely internal, no REPL tool needed
    # (the budget controller hooks into the iteration loop; for the harness we
    #  just note that it is requested and let the ERLM wrapper handle it)

    # O3: Async batched subcall system-prompt addon
    if flags.get("o3"):
        from optimisations.async_subcall import AsyncSubcallManager  # type: ignore
        mgr = AsyncSubcallManager(max_workers=8)
        custom_system_prompt = mgr.get_system_prompt_addon()

    rlm = RLM(
        backend=backend,
        backend_kwargs=backend_kwargs,
        max_depth=1,
        max_iterations=30,
        custom_system_prompt=custom_system_prompt,
        custom_tools=custom_tools if custom_tools else None,
    )
    rlm._erlm_flags = flags  # stash for later use in run_method
    return rlm


# ---------------------------------------------------------------------------
# Method runner
# ---------------------------------------------------------------------------


def run_method(
    method: str,
    sample: Any,
    client: BaseLM,
    erlm_flags: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Run a single *method* on a single *sample* and return a result dict.

    Parameters
    ----------
    method:
        One of the keys in :data:`ALL_METHODS`.
    sample:
        A dataset sample dataclass (CodeQASample, BrowseCompSample, or OolongSample).
    client:
        Pre-constructed LLM client.
    erlm_flags:
        Boolean flags for ERLM optimisations: ``{"o1": True, "o2": False, ...}``

    Returns
    -------
    dict with keys:
        method, sample_id, prediction, ground_truth,
        total_tokens, llm_calls, wall_clock, truncated
    """
    if erlm_flags is None:
        erlm_flags = {}

    sample_id = _sample_id(sample)
    document = _sample_document(sample)
    question = _sample_question(sample)
    ground_truth = _sample_answer(sample)

    base_result: dict[str, Any] = {
        "method": method,
        "sample_id": sample_id,
        "prediction": "",
        "ground_truth": ground_truth,
        "total_tokens": 0,
        "llm_calls": 0,
        "wall_clock": 0.0,
        "truncated": False,
    }

    try:
        if method == "vanilla":
            baseline = VanillaBaseline(client=client)
            result = baseline.run(document=document, question=question)

        elif method == "compaction":
            baseline = CompactionBaseline(client=client)
            result = baseline.run(document=document, question=question)

        elif method == "react":
            baseline = ReActBaseline(client=client)
            result = baseline.run(document=document, question=question)

        elif method == "rlm":
            # Use the BASELINE RLM directly
            backend = "gemini" if isinstance(client, GeminiClient) else "openai"
            backend_kwargs: dict[str, Any] = {"model_name": client.model_name}
            if backend == "openai":
                base_url = getattr(client, "base_url", None)
                if base_url:
                    backend_kwargs["base_url"] = base_url

            rlm = RLM(
                backend=backend,
                backend_kwargs=backend_kwargs,
                max_depth=1,
                max_iterations=30,
            )
            t0 = time.perf_counter()
            prompt = (
                f"Answer the following question based on the provided document.\n\n"
                f"Document:\n{document}\n\nQuestion: {question}\n\nAnswer:"
            )
            rlm_result = rlm.completion(prompt)
            elapsed = time.perf_counter() - t0

            # Extract answer string from result
            prediction = ""
            if hasattr(rlm_result, "final_answer") and rlm_result.final_answer:
                prediction = str(rlm_result.final_answer)
            elif hasattr(rlm_result, "stdout"):
                prediction = str(rlm_result.stdout).strip()
            else:
                prediction = str(rlm_result).strip()

            # Token / call counts from the RLM result
            total_tokens = 0
            llm_calls = 0
            if hasattr(rlm_result, "rlm_calls"):
                llm_calls = len(rlm_result.rlm_calls)
                for call in rlm_result.rlm_calls:
                    if hasattr(call, "usage_summary"):
                        total_tokens += (
                            call.usage_summary.total_input_tokens
                            + call.usage_summary.total_output_tokens
                        )

            result = BaselineResult(
                method="rlm",
                answer=prediction,
                total_tokens=total_tokens,
                llm_calls=max(llm_calls, 1),
                wall_clock_seconds=elapsed,
                truncated=False,
            )

        elif method.startswith("erlm"):
            # Determine flags from the method name
            active_flags: dict[str, bool] = {}
            if method == "erlm_o1":
                active_flags = {"o1": True, "o2": False, "o3": False}
            elif method == "erlm_o2":
                active_flags = {"o1": False, "o2": True, "o3": False}
            elif method == "erlm_o3":
                active_flags = {"o1": False, "o2": False, "o3": True}
            elif method == "erlm_o1o2o3":
                active_flags = {"o1": True, "o2": True, "o3": True}
            elif method == "erlm_all":
                active_flags = {"o1": True, "o2": True, "o3": True, "o4": True, "o5": True}
            else:
                active_flags = erlm_flags  # caller-specified flags

            erlm = _build_erlm(client=client, flags=active_flags)

            # Handle PromptIndexer wrapping for O1 (index the document at run time)
            prompt = (
                f"Answer the following question based on the provided document.\n\n"
                f"Document:\n{document}\n\nQuestion: {question}\n\nAnswer:"
            )

            custom_tools: dict[str, Any] | None = None
            if active_flags.get("o1"):
                from optimisations.prompt_indexer import PromptIndexer  # type: ignore
                indexer = PromptIndexer(chunk_size=2000, overlap=200, top_k=5)
                indexer.build_index(document)
                custom_tools = indexer.get_custom_tool()
                # Re-inject the now-populated tool if the rlm supports it
                if hasattr(erlm, "custom_tools"):
                    erlm.custom_tools = custom_tools

            t0 = time.perf_counter()
            erlm_result = erlm.completion(prompt)
            elapsed = time.perf_counter() - t0

            prediction = ""
            if hasattr(erlm_result, "final_answer") and erlm_result.final_answer:
                prediction = str(erlm_result.final_answer)
            elif hasattr(erlm_result, "stdout"):
                prediction = str(erlm_result.stdout).strip()
            else:
                prediction = str(erlm_result).strip()

            total_tokens = 0
            llm_calls = 0
            if hasattr(erlm_result, "rlm_calls"):
                llm_calls = len(erlm_result.rlm_calls)
                for call in erlm_result.rlm_calls:
                    if hasattr(call, "usage_summary"):
                        total_tokens += (
                            call.usage_summary.total_input_tokens
                            + call.usage_summary.total_output_tokens
                        )

            result = BaselineResult(
                method=method,
                answer=prediction,
                total_tokens=total_tokens,
                llm_calls=max(llm_calls, 1),
                wall_clock_seconds=elapsed,
                truncated=False,
            )

        else:
            raise ValueError(f"Unknown method '{method}'")

    except Exception as exc:
        tb = traceback.format_exc()
        base_result["prediction"] = f"ERROR: {exc}"
        base_result["_traceback"] = tb
        return base_result

    base_result["prediction"] = result.answer
    base_result["total_tokens"] = result.total_tokens
    base_result["llm_calls"] = result.llm_calls
    base_result["wall_clock"] = result.wall_clock_seconds
    base_result["truncated"] = result.truncated
    return base_result


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Lower-case, strip punctuation and extra whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def _token_overlap_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between *prediction* and *ground_truth*."""
    pred_tokens = _normalize(prediction).split()
    gt_tokens = _normalize(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return 0.0

    pred_set = set(pred_tokens)
    gt_set = set(gt_tokens)
    common = pred_set & gt_set

    if not common:
        return 0.0

    precision = len(common) / len(pred_set)
    recall = len(common) / len(gt_set)
    return 2 * precision * recall / (precision + recall)


def score_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add scoring metrics to each result in-place and return the list.

    Metrics added per result:
        exact_match : 1.0 if normalized prediction == normalized ground truth
        f1          : token-level F1 score
        contains    : 1.0 if ground_truth is a substring of prediction

    Parameters
    ----------
    results:
        List of result dicts as produced by :func:`run_method`.

    Returns
    -------
    list[dict]
        The same list with ``exact_match``, ``f1``, and ``contains`` fields added.
    """
    for r in results:
        prediction = r.get("prediction", "")
        ground_truth = r.get("ground_truth", "")

        # Skip error predictions gracefully
        if prediction.startswith("ERROR:"):
            r["exact_match"] = 0.0
            r["f1"] = 0.0
            r["contains"] = 0.0
            continue

        norm_pred = _normalize(prediction)
        norm_gt = _normalize(ground_truth)

        r["exact_match"] = 1.0 if norm_pred == norm_gt else 0.0
        r["f1"] = _token_overlap_f1(prediction, ground_truth)
        r["contains"] = 1.0 if norm_gt and norm_gt in norm_pred else 0.0

    return results


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def _build_summary(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate per-method metrics across all result dicts.

    Returns
    -------
    list of dicts, one per unique method, with keys:
        method, n_samples, avg_f1, avg_exact, avg_contains,
        avg_tokens, avg_llm_calls, avg_time, error_rate
    """
    from collections import defaultdict

    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        buckets[r["method"]].append(r)

    summary_rows = []
    for method, rows in sorted(buckets.items()):
        n = len(rows)
        errors = sum(1 for r in rows if str(r.get("prediction", "")).startswith("ERROR:"))

        def _avg(key: str) -> float:
            vals = [r.get(key, 0.0) for r in rows if not str(r.get("prediction", "")).startswith("ERROR:")]
            return sum(vals) / len(vals) if vals else 0.0

        summary_rows.append({
            "method": method,
            "n_samples": n,
            "avg_f1": round(_avg("f1"), 4),
            "avg_exact": round(_avg("exact_match"), 4),
            "avg_contains": round(_avg("contains"), 4),
            "avg_tokens": round(_avg("total_tokens"), 1),
            "avg_llm_calls": round(_avg("llm_calls"), 2),
            "avg_time": round(_avg("wall_clock"), 3),
            "error_rate": round(errors / n, 4) if n else 0.0,
        })

    return summary_rows


def _print_summary_table(summary_rows: list[dict[str, Any]]) -> None:
    """Print a formatted summary table to stdout."""
    if not summary_rows:
        print("No results to summarise.")
        return

    header = f"{'Method':<20} {'avg_f1':>8} {'avg_exact':>10} {'avg_contains':>13} {'avg_tokens':>11} {'avg_time(s)':>12} {'errors':>7}"
    print()
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            f"{row['method']:<20} "
            f"{row['avg_f1']:>8.4f} "
            f"{row['avg_exact']:>10.4f} "
            f"{row['avg_contains']:>13.4f} "
            f"{row['avg_tokens']:>11.1f} "
            f"{row['avg_time']:>12.3f} "
            f"{row['error_rate']:>7.4f}"
        )
    print("=" * len(header))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RLM vs ERLM evaluation harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="codeqa",
        help=f"Benchmark to evaluate on. One of: {ALL_BENCHMARKS + ['all']}",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        help=f"Method to run. One of: {ALL_METHODS + ['all']}",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Maximum number of samples to evaluate per benchmark.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini",
        help="Model shorthand: gemini | ollama | vllm | openai",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Specific model identifier passed to the provider API.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=_BENCHMARKS_DIR,
        help="Directory where JSONL results and CSV summaries are saved.",
    )
    parser.add_argument(
        "--erlm_flags",
        type=str,
        default="",
        help="Comma-separated ERLM optimisation flags, e.g. 'o1,o2,o3'.",
    )

    args = parser.parse_args()

    # Resolve benchmarks to run
    benchmarks_to_run: list[str]
    if args.benchmark == "all":
        benchmarks_to_run = ALL_BENCHMARKS
    else:
        if args.benchmark not in ALL_BENCHMARKS:
            parser.error(f"Unknown benchmark '{args.benchmark}'. Choose from: {ALL_BENCHMARKS + ['all']}")
        benchmarks_to_run = [args.benchmark]

    # Resolve methods to run
    methods_to_run: list[str]
    if args.method == "all":
        methods_to_run = ALL_METHODS
    else:
        if args.method not in ALL_METHODS:
            parser.error(f"Unknown method '{args.method}'. Choose from: {ALL_METHODS + ['all']}")
        methods_to_run = [args.method]

    # Parse erlm_flags into a dict
    erlm_flags: dict[str, bool] = {}
    if args.erlm_flags:
        for flag in args.erlm_flags.split(","):
            flag = flag.strip()
            if flag:
                erlm_flags[flag] = True

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build the client once (shared across all runs)
    print(f"Building client: model={args.model}, model_name={args.model_name}", flush=True)
    client = get_client(model=args.model, model_name=args.model_name)

    all_results: list[dict[str, Any]] = []

    for benchmark in benchmarks_to_run:
        print(f"\nLoading dataset: {benchmark} (n_samples={args.n_samples})", flush=True)
        try:
            samples = get_dataset(benchmark=benchmark, n_samples=args.n_samples)
        except Exception as exc:
            print(f"[ERROR] Failed to load dataset '{benchmark}': {exc}", flush=True)
            continue

        print(f"  Loaded {len(samples)} samples.", flush=True)

        for method in methods_to_run:
            print(f"\n  Running method: {method} on {benchmark}", flush=True)
            method_results: list[dict[str, Any]] = []

            iterator = _progress(
                iterable=samples,
                desc=f"{method}/{benchmark}",
                total=len(samples),
            )

            for sample in iterator:
                result = run_method(
                    method=method,
                    sample=sample,
                    client=client,
                    erlm_flags=erlm_flags,
                )
                result["benchmark"] = benchmark
                method_results.append(result)

            # Score this method's results
            method_results = score_results(method_results)
            all_results.extend(method_results)

            # Save per-method JSONL
            safe_model = args.model.replace("/", "_")
            safe_model_name = (args.model_name or "").replace("/", "_")
            suffix = f"_{safe_model_name}" if safe_model_name else ""
            out_filename = f"{benchmark}_{method}_{safe_model}{suffix}_{timestamp}.jsonl"
            out_path = os.path.join(args.output_dir, out_filename)

            with open(out_path, "w", encoding="utf-8") as fh:
                for r in method_results:
                    fh.write(json.dumps(r, default=str) + "\n")

            print(f"  Saved {len(method_results)} results → {out_path}", flush=True)

    if not all_results:
        print("No results collected. Exiting.", flush=True)
        return

    # Score all results (in case some were collected without scoring)
    all_results = score_results(all_results)

    # Build summary
    summary_rows = _build_summary(all_results)
    _print_summary_table(summary_rows)

    # Save combined JSONL
    combined_benchmarks = "_".join(benchmarks_to_run)
    combined_methods = "_".join(methods_to_run) if len(methods_to_run) <= 3 else "multi"
    safe_model = args.model.replace("/", "_")
    combined_filename = f"combined_{combined_benchmarks}_{combined_methods}_{safe_model}_{timestamp}.jsonl"
    combined_path = os.path.join(args.output_dir, combined_filename)
    with open(combined_path, "w", encoding="utf-8") as fh:
        for r in all_results:
            fh.write(json.dumps(r, default=str) + "\n")
    print(f"Combined JSONL → {combined_path}", flush=True)

    # Save summary CSV
    csv_filename = f"summary_{combined_benchmarks}_{combined_methods}_{safe_model}_{timestamp}.csv"
    csv_path = os.path.join(args.output_dir, csv_filename)
    if summary_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Summary CSV     → {csv_path}", flush=True)


if __name__ == "__main__":
    main()
