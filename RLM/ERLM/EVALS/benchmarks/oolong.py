from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterator

try:
    from datasets import load_dataset
except (ImportError, AttributeError) as exc:
    raise ImportError(
        "The `datasets` library is required to load the OOLONG dataset. "
        "Install it with: pip install datasets"
    ) from exc

# HuggingFace repo.  Actual available configs: ['dnd', 'toy_dnd'].
# 'dnd' is the full benchmark (6072 test rows, median doc = 1.2M chars).
_HF_DATASET = "oolongbench/oolong-real"
_HF_CONFIG = "dnd"

# Qwen3-Coder-480B-A35B context window: 128K tokens ≈ 512,000 chars.
# Only keep docs that overflow the context so RLM is genuinely required.
_CONTEXT_WINDOW_CHARS: int = 512_000


@dataclass
class OolongSample:
    """A single sample from the OOLONG benchmark.

    Attributes:
        id: Unique identifier for the sample.
        document: The long document used as context.
        question: Aggregation or multi-hop reasoning question.
        answer: Ground-truth answer string.
        task_type: Type of reasoning required, e.g. "aggregation", "multi_hop".
    """

    id: str
    document: str
    question: str
    answer: str
    task_type: str


class OolongDataset:
    """Dataset loader for the OOLONG benchmark (oolongbench/oolong-real).

    Loads the ``trec_coarse`` configuration by default (falls back to the
    default config when that is unavailable).  Column names are resolved
    defensively to handle schema differences between configs.

    Args:
        max_samples: Maximum number of samples to return after shuffling.
        seed: Random seed used for shuffling before sampling.
    """

    def __init__(
        self,
        max_samples_per_type: int = 50,  # stratified: 50 per question_type
        min_doc_length: int = _CONTEXT_WINDOW_CHARS,
        seed: int = 42,
    ) -> None:
        self.max_samples_per_type = max_samples_per_type
        self.min_doc_length = min_doc_length
        self.seed = seed
        self._samples: list[OolongSample] = []

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

    def load(self) -> list[OolongSample]:
        """Load samples from HuggingFace.

        Tries the ``trec_coarse`` config first; falls back to no-config (default)
        if that raises an error.

        Returns:
            A list of up to ``max_samples`` :class:`OolongSample` objects.
        """
        raw = self._load_raw()
        columns = set(raw.column_names)

        # OOLONG 'dnd' config actual columns:
        # id, context_window_id, context_window_text, question, answer,
        # question_type, episodes, campaign
        doc_candidates = ["context_window_text", "document", "context", "passage", "text", "input_text"]
        q_candidates = ["question", "query", "input", "problem"]
        ans_candidates = ["answer", "answers", "gold_answer", "label", "output"]
        task_candidates = ["question_type", "task_type", "task", "type", "category", "domain"]
        id_candidates = ["id", "context_window_id", "_id", "idx", "sample_id", "qid"]

        # Warn if core fields are missing
        for label, candidates in [
            ("document/context", doc_candidates),
            ("question/query", q_candidates),
            ("answer", ans_candidates),
        ]:
            if not any(c in columns for c in candidates):
                warnings.warn(
                    f"OOLONG: could not find a '{label}' column. "
                    f"Available columns: {sorted(columns)}. "
                    "Field will default to empty string.",
                    stacklevel=2,
                )

        samples: list[OolongSample] = []
        for idx, row in enumerate(raw):
            document = self._pick(row, doc_candidates)
            # Skip docs that fit within the model context — RLM adds no value there
            if len(document) < self.min_doc_length:
                continue

            raw_answer = None
            for key in ans_candidates:
                if key in row and row[key] is not None:
                    raw_answer = row[key]
                    break
            if isinstance(raw_answer, list):
                answer = raw_answer[0] if raw_answer else ""
            else:
                answer = str(raw_answer) if raw_answer is not None else ""

            sample = OolongSample(
                id=self._pick(row, id_candidates, fallback=str(idx)),
                document=document,
                question=self._pick(row, q_candidates),
                answer=answer,
                task_type=self._pick(row, task_candidates, fallback="unknown"),
            )
            samples.append(sample)

        # Stratified sample: equal representation across question_type values
        import random
        from collections import defaultdict
        rng = random.Random(self.seed)
        by_type: dict[str, list[OolongSample]] = defaultdict(list)
        for s in samples:
            by_type[s.task_type].append(s)

        stratified: list[OolongSample] = []
        for type_samples in by_type.values():
            rng.shuffle(type_samples)
            stratified.extend(type_samples[: self.max_samples_per_type])

        rng.shuffle(stratified)
        self._samples = stratified
        return self._samples

    def _load_raw(self):
        """Load the HuggingFace dataset.

        Primary: config='dnd', split='test' (6072 rows, median doc 1.2M chars).
        Falls back through 'validation', 'train', and then toy_dnd if needed.
        """
        for config in [_HF_CONFIG, "toy_dnd"]:
            for split in ["test", "validation", "train"]:
                try:
                    return load_dataset(_HF_DATASET, config, split=split)
                except Exception:
                    continue

        # Last resort: no config
        try:
            raw = load_dataset(_HF_DATASET)
            split_name = list(raw.keys())[0]
            warnings.warn(
                f"OOLONG: fell back to default config, split='{split_name}'.",
                stacklevel=3,
            )
            return raw[split_name]
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load {_HF_DATASET}. "
                "Check your internet connection and HuggingFace access."
            ) from exc

    def get_stats(self) -> dict:
        """Return descriptive statistics over the loaded samples.

        Returns:
            A dictionary with keys ``n_samples``, ``avg_doc_length``, and
            ``task_type_distribution``.
        """
        if not self._samples:
            return {"n_samples": 0, "avg_doc_length": 0.0, "task_type_distribution": {}}

        avg_len = sum(len(s.document) for s in self._samples) / len(self._samples)
        task_dist: dict[str, int] = {}
        for s in self._samples:
            task_dist[s.task_type] = task_dist.get(s.task_type, 0) + 1

        return {
            "n_samples": len(self._samples),
            "avg_doc_length": avg_len,
            "task_type_distribution": task_dist,
        }

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[OolongSample]:
        return iter(self._samples)

    def __getitem__(self, index: int) -> OolongSample:
        return self._samples[index]


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading OOLONG samples …")
    ds = OolongDataset(max_samples=3)
    samples = ds.load()
    print(f"Loaded {len(samples)} sample(s)\n")
    for i, s in enumerate(samples):
        print(f"--- Sample {i} ---")
        print(f"  id        : {s.id}")
        print(f"  task_type : {s.task_type}")
        print(f"  question  : {s.question[:120]}")
        print(f"  answer    : {s.answer[:80]}")
        print(f"  document  : {s.document[:200]} …")
        print()
    print("Stats:", ds.get_stats())
