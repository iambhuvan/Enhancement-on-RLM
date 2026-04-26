"""
Compaction Baseline
===================
Split the document into fixed-size chunks, extract question-relevant
information from each chunk in parallel (up to ``max_workers`` threads),
then answer from the concatenated extracts in a final LLM call.

Pipeline
--------
1. Split document into chunks of ``chunk_size`` characters.
2. For each chunk, call the LLM to extract relevant information (parallel).
3. Filter out 'NONE' responses; concatenate remaining extracts.
4. Call the LLM once more to produce the final answer from the extracts.
"""

from __future__ import annotations

import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'BASELINE'))

from rlm.clients.base_lm import BaseLM
from .vanilla import BaselineResult


class CompactionBaseline:
    """Chunk-extract-then-answer baseline with parallel extraction.

    Parameters
    ----------
    client:
        Any :class:`rlm.clients.base_lm.BaseLM` instance.
    chunk_size:
        Number of characters per chunk.  Default: 4 000.
    max_workers:
        Maximum concurrent threads for the parallel extraction step.
        Default: 8.
    """

    def __init__(
        self,
        client: BaseLM,
        chunk_size: int = 4_000,
        max_workers: int = 8,
    ) -> None:
        self.client = client
        self.chunk_size = chunk_size
        self.max_workers = max_workers

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_chunks(self, document: str) -> list[str]:
        """Split *document* into non-overlapping chunks of ``chunk_size`` chars."""
        return [
            document[i : i + self.chunk_size]
            for i in range(0, len(document), self.chunk_size)
        ]

    def _extract_from_chunk(self, chunk: str, question: str) -> str:
        """Call the LLM to extract information relevant to *question* from *chunk*.

        Returns the model response string.
        """
        prompt = (
            f"Extract any information relevant to answering this question: '{question}'\n\n"
            f"Text:\n{chunk}\n\n"
            f"Relevant info (or 'NONE' if not relevant):"
        )
        return self.client.completion(prompt)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, document: str, question: str) -> BaselineResult:
        """Run the compaction baseline on a single (document, question) pair.

        Parameters
        ----------
        document:
            The full document text.
        question:
            The question to answer from the document.

        Returns
        -------
        BaselineResult
            Result with ``method="compaction"`` and
            ``llm_calls = n_chunks + 1``.
        """
        start = time.perf_counter()
        total_tokens = 0

        # Step 1: split into chunks
        chunks = self._split_chunks(document)
        n_chunks = len(chunks)

        # Step 2: parallel extraction
        extracts: list[Optional[str]] = [None] * n_chunks

        def _extract_indexed(idx: int, chunk: str) -> tuple[int, str]:
            return idx, self._extract_from_chunk(chunk, question)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_extract_indexed, i, chunk): i
                for i, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                idx, response = future.result()
                extracts[idx] = response

        # Step 3: filter NONE responses and accumulate token estimates
        relevant: list[str] = []
        for response in extracts:
            if response is None:
                continue
            stripped = response.strip()
            if stripped.upper() == "NONE" or stripped == "":
                continue
            relevant.append(stripped)

        # Attempt to read token usage from the client (best-effort)
        try:
            usage_summary = self.client.get_usage_summary()
            for model_usage in usage_summary.model_usage_summaries.values():
                total_tokens += (
                    model_usage.total_input_tokens + model_usage.total_output_tokens
                )
        except Exception:
            # Heuristic fallback: ~4 chars per token across all chunk calls
            for chunk in chunks:
                total_tokens += len(chunk) // 4

        # Step 4: final synthesis call
        concatenated_extracts = "\n\n".join(relevant) if relevant else "No relevant information found."

        final_prompt = (
            f"Answer this question based on the extracted information:\n\n"
            f"Question: {question}\n\n"
            f"Extracted info:\n{concatenated_extracts}\n\n"
            f"Answer:"
        )
        final_answer = self.client.completion(final_prompt)

        # Add tokens for the final synthesis call
        try:
            last_usage = self.client.get_last_usage()
            total_tokens += last_usage.total_input_tokens + last_usage.total_output_tokens
        except Exception:
            total_tokens += (len(final_prompt) + len(final_answer)) // 4

        elapsed = time.perf_counter() - start

        return BaselineResult(
            method="compaction",
            answer=final_answer.strip(),
            total_tokens=total_tokens,
            llm_calls=n_chunks + 1,
            wall_clock_seconds=elapsed,
            truncated=False,
        )
