from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Iterator

try:

    import importlib as _il
    _hf = _il.import_module("datasets")
    load_dataset = _hf.load_dataset
except (ImportError, AttributeError) as exc:
    raise ImportError(
        "The `datasets` library is required to load LongBench-v2. "
        "Install it with: pip install datasets"
    ) from exc


@dataclass
class CodeQASample:
    """A single sample from the LongBench-v2 CodeQA subset.

    Attributes:
        id: Unique identifier for the sample.
        document: The long code context (may span a multi-file repo).
        question: The question posed about the code context.
        answer: Ground-truth answer string.
        length: Document length in characters.
        difficulty: Difficulty tier — "easy", "medium", or "hard".
    """

    id: str
    document: str
    question: str
    answer: str
    length: int
    difficulty: str


class CodeQADataset:
    """Dataset loader for the CodeQA subset of LongBench-v2.

    The underlying HuggingFace dataset is ``zai-org/LongBench-v2`` (test split).
    Rows where the domain is ``"code"`` or the context length satisfies the
    configured bounds are retained.

    Args:
        max_samples: Maximum number of samples to return after filtering.
        min_doc_length: Minimum document length in characters (inclusive).
        max_doc_length: Maximum document length in characters (inclusive).
        seed: Random seed used for shuffling before sampling.
    """

    _HF_DATASET = "zai-org/LongBench-v2"
    # The dataset only has a 'train' split despite being an evaluation benchmark.
    _SPLIT = "train"

    # Qwen3-Coder-480B-A35B has a 128K token context window.
    # 128K tokens × ~4 chars/token = 512,000 chars.
    # We only keep docs that EXCEED the model context so RLM is actually needed.
    _CONTEXT_WINDOW_CHARS: int = 512_000

    def __init__(
        self,
        max_samples: int | None = None,  # None = use ALL qualifying samples
        min_doc_length: int = _CONTEXT_WINDOW_CHARS,
        max_doc_length: int = 10_000_000,
        domains: list[str] | None = None,  # None = all domains
        seed: int = 42,
    ) -> None:
        self.max_samples = max_samples
        self.min_doc_length = min_doc_length
        self.max_doc_length = max_doc_length
        # Default: focus on Code Repository and Multi-Document QA
        self.domains = domains or [
            "Code Repository Understanding",
            "Multi-Document QA",
            "Single-Document QA",
        ]
        self.seed = seed
        self._samples: list[CodeQASample] = []

    # ------------------------------------------------------------------
    # Column resolution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_column(row: dict, candidates: list[str], fallback: str = "") -> str:
        """Return the value of the first matching candidate column."""
        for col in candidates:
            if col in row and row[col] is not None:
                return str(row[col])
        return fallback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> list[CodeQASample]:
        """Load and filter samples from HuggingFace.

        Returns:
            A list of up to ``max_samples`` :class:`CodeQASample` objects.
        """
        raw = load_dataset(self._HF_DATASET, split=self._SPLIT)

        columns = set(raw.column_names)

        # Identify column names defensively
        id_col = next((c for c in ["_id", "id"] if c in columns), None)
        ctx_col = next((c for c in ["context", "document", "passage"] if c in columns), None)
        q_col = next((c for c in ["input", "question", "query"] if c in columns), None)
        ans_col = next((c for c in ["answers", "answer", "gold_answer"] if c in columns), None)
        len_col = next((c for c in ["length", "doc_length"] if c in columns), None)
        diff_col = next((c for c in ["difficulty", "level"] if c in columns), None)
        domain_col = next((c for c in ["domain", "type", "task_type"] if c in columns), None)

        if ctx_col is None:
            warnings.warn(
                "Could not find a context/document column in LongBench-v2. "
                f"Available columns: {sorted(columns)}. Attempting to continue.",
                stacklevel=2,
            )

        samples: list[CodeQASample] = []
        for row in raw:
            # Filter by domain
            if domain_col is not None and self.domains:
                domain_val = str(row.get(domain_col, ""))
                if domain_val not in self.domains:
                    continue

            document: str = self._resolve_column(row, [ctx_col] if ctx_col else [])
            # Always use actual character length for filtering — the 'length'
            # column in LongBench-v2 contains category labels ("short"/"medium"/"long"),
            # not integer token/char counts.
            doc_len: int = len(document)

            if doc_len < self.min_doc_length or doc_len > self.max_doc_length:
                continue

            # Answer: LongBench-v2 uses a list in 'answers'
            raw_answer = row.get(ans_col, "") if ans_col else ""
            if isinstance(raw_answer, list):
                answer = raw_answer[0] if raw_answer else ""
            else:
                answer = str(raw_answer) if raw_answer is not None else ""

            sample = CodeQASample(
                id=self._resolve_column(row, [id_col] if id_col else [], fallback=str(len(samples))),
                document=document,
                question=self._resolve_column(row, [q_col] if q_col else []),
                answer=answer,
                length=doc_len,
                difficulty=self._resolve_column(row, [diff_col] if diff_col else [], fallback="unknown"),
            )
            samples.append(sample)

        # Shuffle deterministically, then cap only if max_samples is set
        import random
        rng = random.Random(self.seed)
        rng.shuffle(samples)
        self._samples = samples if self.max_samples is None else samples[: self.max_samples]
        return self._samples

    def get_stats(self) -> dict:
        """Return descriptive statistics over the loaded samples.

        Returns:
            A dictionary with keys ``n_samples``, ``avg_doc_length``, and
            ``difficulty_distribution``.
        """
        if not self._samples:
            return {"n_samples": 0, "avg_doc_length": 0.0, "difficulty_distribution": {}}

        avg_len = sum(s.length for s in self._samples) / len(self._samples)
        dist: dict[str, int] = {}
        for s in self._samples:
            dist[s.difficulty] = dist.get(s.difficulty, 0) + 1

        return {
            "n_samples": len(self._samples),
            "avg_doc_length": avg_len,
            "difficulty_distribution": dist,
        }

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[CodeQASample]:
        return iter(self._samples)

    def __getitem__(self, index: int) -> CodeQASample:
        return self._samples[index]


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading CodeQA samples from LongBench-v2 …")
    ds = CodeQADataset(max_samples=3, min_doc_length=1_000)
    samples = ds.load()
    print(f"Loaded {len(samples)} sample(s)\n")
    for i, s in enumerate(samples):
        print(f"--- Sample {i} ---")
        print(f"  id        : {s.id}")
        print(f"  difficulty: {s.difficulty}")
        print(f"  length    : {s.length}")
        print(f"  question  : {s.question[:120]}")
        print(f"  answer    : {s.answer[:80]}")
        print(f"  document  : {s.document[:200]} …")
        print()
    print("Stats:", ds.get_stats())
