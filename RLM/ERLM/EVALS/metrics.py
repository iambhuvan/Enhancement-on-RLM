from __future__ import annotations

import re
import statistics
import string
from collections import Counter
from typing import Callable


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

_ARTICLES = frozenset({"a", "an", "the"})
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Normalize a string for lenient string comparison.

    Steps applied in order:
    1. Lowercase.
    2. Strip leading/trailing whitespace.
    3. Remove punctuation.
    4. Remove articles ("a", "an", "the").
    5. Collapse extra internal whitespace.

    Args:
        text: Raw text to normalize.

    Returns:
        Normalized text string.
    """
    if not text:
        return ""
    text = text.lower().strip()
    text = text.translate(_PUNCT_TABLE)
    tokens = [tok for tok in text.split() if tok not in _ARTICLES]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


def exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact-match score after normalization.

    Args:
        prediction: Model output string.
        ground_truth: Reference answer string.

    Returns:
        1.0 if the normalized strings are identical, else 0.0.
    """
    if not prediction or not ground_truth:
        return 0.0
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def contains_match(prediction: str, ground_truth: str) -> float:
    """Check whether the normalized ground truth appears in the normalized prediction.

    Args:
        prediction: Model output string.
        ground_truth: Reference answer string.

    Returns:
        1.0 if the normalized ground truth is a substring of the normalized
        prediction, else 0.0.
    """
    if not prediction or not ground_truth:
        return 0.0
    return 1.0 if normalize_answer(ground_truth) in normalize_answer(prediction) else 0.0


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score.

    Token overlap is computed on normalized token bags.

    Precision = |pred_tokens ∩ gt_tokens| / |pred_tokens|
    Recall    = |pred_tokens ∩ gt_tokens| / |gt_tokens|
    F1        = 2 * P * R / (P + R)

    Args:
        prediction: Model output string.
        ground_truth: Reference answer string.

    Returns:
        F1 score in [0.0, 1.0].
    """
    if not prediction or not ground_truth:
        return 0.0

    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)

    common: int = sum((pred_counter & gt_counter).values())

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gt_tokens)
    return 2.0 * precision * recall / (precision + recall)


def list_recall(prediction: str, ground_truth: str) -> float:
    """Compute recall over a comma/semicolon-delimited list of ground truth items.

    The ground truth is split on commas and semicolons into individual items.
    Each item is normalized, and the fraction of items whose normalized form
    appears in the normalized prediction is returned.

    Args:
        prediction: Model output string.
        ground_truth: Reference answer string (items separated by "," or ";").

    Returns:
        Fraction of ground truth items found in the prediction, in [0.0, 1.0].
    """
    if not prediction or not ground_truth:
        return 0.0

    items = re.split(r"[,;]", ground_truth)
    items = [normalize_answer(item) for item in items if item.strip()]

    if not items:
        return 0.0

    norm_pred = normalize_answer(prediction)
    found = sum(1 for item in items if item and item in norm_pred)
    return found / len(items)


def token_efficiency(tokens_used: int, max_tokens: int) -> float:
    """Measure how efficiently tokens were used relative to a budget.

    A higher score means fewer tokens were consumed relative to the budget.

    Args:
        tokens_used: Number of tokens actually consumed.
        max_tokens: Maximum allowed tokens (budget).

    Returns:
        ``1.0 - (tokens_used / max_tokens)`` clamped to [0.0, 1.0].
        Returns 0.0 if ``max_tokens`` is zero or negative.
    """
    if max_tokens <= 0:
        return 0.0
    ratio = tokens_used / max_tokens
    return max(0.0, min(1.0, 1.0 - ratio))


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

_METRIC_REGISTRY: dict[str, Callable[[str, str], float]] = {
    "exact_match": exact_match,
    "exact": exact_match,
    "f1": f1_score,
    "f1_score": f1_score,
    "contains": contains_match,
    "contains_match": contains_match,
    "list_recall": list_recall,
}


# ---------------------------------------------------------------------------
# MetricsSuite
# ---------------------------------------------------------------------------


class MetricsSuite:
    """A configurable collection of evaluation metrics.

    Args:
        metrics: List of metric names to include.  Supported names are
            ``"exact_match"`` (alias ``"exact"``), ``"f1"`` / ``"f1_score"``,
            ``"contains"`` / ``"contains_match"``, and ``"list_recall"``.

    Raises:
        ValueError: If any name in *metrics* is not recognized.
    """

    def __init__(self, metrics: list[str] = ("exact_match", "f1", "contains")) -> None:
        unknown = [m for m in metrics if m not in _METRIC_REGISTRY]
        if unknown:
            raise ValueError(
                f"Unknown metric(s): {unknown}. "
                f"Available: {sorted(_METRIC_REGISTRY)}"
            )
        self.metrics = list(metrics)
        self._fns: list[tuple[str, Callable[[str, str], float]]] = [
            (name, _METRIC_REGISTRY[name]) for name in self.metrics
        ]

    def score(self, prediction: str, ground_truth: str) -> dict[str, float]:
        """Run all configured metrics on a single prediction/ground-truth pair.

        Args:
            prediction: Model output string.
            ground_truth: Reference answer string.

        Returns:
            Dictionary mapping each metric name to its score in [0.0, 1.0].
        """
        return {name: fn(prediction, ground_truth) for name, fn in self._fns}

    def aggregate(self, results: list[dict[str, float]]) -> dict[str, float]:
        """Aggregate per-sample score dicts into means and standard deviations.

        Args:
            results: List of score dictionaries as returned by :meth:`score`.

        Returns:
            Dictionary with keys ``"<metric>_mean"`` and ``"<metric>_std"`` for
            each metric, plus ``"n_samples"``.  Returns zeros when *results* is
            empty.
        """
        if not results:
            agg: dict[str, float] = {"n_samples": 0.0}
            for name in self.metrics:
                agg[f"{name}_mean"] = 0.0
                agg[f"{name}_std"] = 0.0
            return agg

        agg = {"n_samples": float(len(results))}
        for name in self.metrics:
            values = [r[name] for r in results if name in r]
            if not values:
                agg[f"{name}_mean"] = 0.0
                agg[f"{name}_std"] = 0.0
            elif len(values) == 1:
                agg[f"{name}_mean"] = values[0]
                agg[f"{name}_std"] = 0.0
            else:
                agg[f"{name}_mean"] = statistics.mean(values)
                agg[f"{name}_std"] = statistics.stdev(values)
        return agg


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    suite = MetricsSuite(metrics=["exact_match", "f1", "contains", "list_recall"])

    test_cases = [
        ("The quick brown fox", "the quick brown fox"),
        ("Paris is the answer", "Paris"),
        ("apples, bananas, and oranges", "apples; bananas"),
        ("completely wrong", "absolutely different"),
        ("", "non-empty ground truth"),
    ]

    results: list[dict[str, float]] = []
    for pred, gt in test_cases:
        scores = suite.score(pred, gt)
        results.append(scores)
        print(f"pred={pred!r:40s}  gt={gt!r:30s}  scores={scores}")

    print("\nAggregated:", suite.aggregate(results))

    print("\ntoken_efficiency(500, 1000):", token_efficiency(500, 1000))
    print("token_efficiency(1200, 1000):", token_efficiency(1200, 1000))
    print("token_efficiency(0, 0):", token_efficiency(0, 0))
