from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterator

try:
    from datasets import load_dataset
except (ImportError, AttributeError) as exc:
    raise ImportError(
        "The `datasets` library is required to load BrowseComp-Plus. "
        "Install it with: pip install datasets"
    ) from exc


@dataclass
class BrowseCompSample:
    """A single sample from the BrowseComp-Plus dataset.

    Attributes:
        id: Unique identifier for the sample.
        problem: The research question or task.
        answer: Ground-truth answer string.
        corpus: Concatenated document corpus (may be very long).
        difficulty: Difficulty tier — "easy", "medium", or "hard".
        topic: Broad subject area, e.g. "science", "history", "tech", "mixed".
    """

    id: str
    problem: str
    answer: str
    corpus: str
    difficulty: str
    topic: str


class BrowseCompDataset:
    """Dataset loader for BrowseComp-Plus.

    Loads ``Tevatron/browsecomp-plus`` from HuggingFace and wraps each row as a
    :class:`BrowseCompSample`.  Column names are resolved defensively to handle
    schema variations across dataset versions.

    Args:
        max_samples: Maximum number of samples to return after filtering.
        difficulty: If given, keep only rows whose difficulty matches this value
            (case-insensitive).  One of ``"easy"``, ``"medium"``, ``"hard"``.
        seed: Random seed used for shuffling before sampling.
    """

    _HF_DATASET = "Tevatron/browsecomp-plus"

    def __init__(
        self,
        max_samples: int = 100,
        difficulty: str | None = None,
        seed: int = 42,
    ) -> None:
        self.max_samples = max_samples
        self.difficulty = difficulty.lower() if difficulty else None
        self.seed = seed
        self._samples: list[BrowseCompSample] = []

    # ------------------------------------------------------------------
    # Column resolution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pick(row: dict, candidates: list[str], fallback: str = "") -> str:
        """Return the string value of the first matching key in *row*."""
        for key in candidates:
            val = row.get(key)
            if val is not None:
                return str(val)
        return fallback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> list[BrowseCompSample]:
        """Load and optionally filter samples from HuggingFace.

        Returns:
            A list of up to ``max_samples`` :class:`BrowseCompSample` objects.
        """
        try:
            raw = load_dataset(self._HF_DATASET, split="train")
        except Exception:
            # Some datasets only have a default split
            try:
                raw = load_dataset(self._HF_DATASET)
                # Take the first available split
                split_name = list(raw.keys())[0]
                raw = raw[split_name]
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load {self._HF_DATASET}. "
                    "Check your internet connection and HuggingFace access."
                ) from exc

        columns = set(raw.column_names)

        # --- Resolve column names ---
        prob_candidates = ["problem", "question", "query", "input"]
        ans_candidates = ["answer", "gold_answer", "answers", "solution"]
        corpus_candidates = ["corpus", "documents", "context", "passage", "document"]
        diff_candidates = ["difficulty", "level", "hardness"]
        topic_candidates = ["topic", "category", "subject", "domain"]
        id_candidates = ["id", "_id", "idx", "sample_id"]

        missing: list[str] = []
        for label, candidates in [
            ("problem/question", prob_candidates),
            ("answer", ans_candidates),
            ("corpus/documents", corpus_candidates),
        ]:
            if not any(c in columns for c in candidates):
                missing.append(label)

        if missing:
            warnings.warn(
                f"BrowseComp-Plus: could not find columns for {missing}. "
                f"Available columns: {sorted(columns)}. "
                "Values will default to empty strings.",
                stacklevel=2,
            )

        samples: list[BrowseCompSample] = []
        for idx, row in enumerate(raw):
            difficulty_val = self._pick(row, diff_candidates, fallback="unknown").lower()

            # Filter by difficulty if requested
            if self.difficulty is not None and difficulty_val != self.difficulty:
                continue

            # Answers may be stored as a list
            raw_answer = None
            for key in ans_candidates:
                if key in row and row[key] is not None:
                    raw_answer = row[key]
                    break
            if isinstance(raw_answer, list):
                answer = raw_answer[0] if raw_answer else ""
            else:
                answer = str(raw_answer) if raw_answer is not None else ""

            sample = BrowseCompSample(
                id=self._pick(row, id_candidates, fallback=str(idx)),
                problem=self._pick(row, prob_candidates),
                answer=answer,
                corpus=self._pick(row, corpus_candidates),
                difficulty=difficulty_val,
                topic=self._pick(row, topic_candidates, fallback="unknown"),
            )
            samples.append(sample)

        # Shuffle deterministically and cap
        import random
        rng = random.Random(self.seed)
        rng.shuffle(samples)
        self._samples = samples[: self.max_samples]
        return self._samples

    def get_stats(self) -> dict:
        """Return descriptive statistics over the loaded samples.

        Returns:
            A dictionary with keys ``n_samples``, ``avg_corpus_length``,
            ``difficulty_distribution``, and ``topic_distribution``.
        """
        if not self._samples:
            return {
                "n_samples": 0,
                "avg_corpus_length": 0.0,
                "difficulty_distribution": {},
                "topic_distribution": {},
            }

        avg_len = sum(len(s.corpus) for s in self._samples) / len(self._samples)
        diff_dist: dict[str, int] = {}
        topic_dist: dict[str, int] = {}
        for s in self._samples:
            diff_dist[s.difficulty] = diff_dist.get(s.difficulty, 0) + 1
            topic_dist[s.topic] = topic_dist.get(s.topic, 0) + 1

        return {
            "n_samples": len(self._samples),
            "avg_corpus_length": avg_len,
            "difficulty_distribution": diff_dist,
            "topic_distribution": topic_dist,
        }

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[BrowseCompSample]:
        return iter(self._samples)

    def __getitem__(self, index: int) -> BrowseCompSample:
        return self._samples[index]


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading BrowseComp-Plus samples …")
    ds = BrowseCompDataset(max_samples=3)
    samples = ds.load()
    print(f"Loaded {len(samples)} sample(s)\n")
    for i, s in enumerate(samples):
        print(f"--- Sample {i} ---")
        print(f"  id        : {s.id}")
        print(f"  difficulty: {s.difficulty}")
        print(f"  topic     : {s.topic}")
        print(f"  problem   : {s.problem[:120]}")
        print(f"  answer    : {s.answer[:80]}")
        print(f"  corpus    : {s.corpus[:200]} …")
        print()
    print("Stats:", ds.get_stats())
