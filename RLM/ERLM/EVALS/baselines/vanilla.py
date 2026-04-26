"""
Vanilla Baseline
================
Single LLM call with the full document + question.
The document is truncated to ``max_chars`` characters when it exceeds that limit.
"""

from __future__ import annotations

import sys
import os
import time
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'BASELINE'))

from rlm.clients.base_lm import BaseLM


@dataclass
class BaselineResult:
    """Result record shared across all baseline methods.

    Attributes:
        method:              Name of the baseline method that produced this result.
        answer:              The predicted answer string.
        total_tokens:        Approximate total tokens used (input + output).
        llm_calls:           Number of LLM API calls made.
        wall_clock_seconds:  Wall-clock elapsed time in seconds.
        truncated:           Whether the document was truncated before the call.
    """

    method: str
    answer: str
    total_tokens: int
    llm_calls: int
    wall_clock_seconds: float
    truncated: bool = False


class VanillaBaseline:
    """Single-shot baseline: feed the whole document and the question to the LLM.

    If the document exceeds *max_chars* it is truncated with a trailing
    ellipsis notice so the model is aware of the truncation.

    Parameters
    ----------
    client:
        Any :class:`rlm.clients.base_lm.BaseLM` instance.
    max_chars:
        Maximum number of characters from the document to include in the
        prompt.  Documents longer than this are truncated.  Default: 400 000.
    """

    def __init__(self, client: BaseLM, max_chars: int = 400_000) -> None:
        self.client = client
        self.max_chars = max_chars

    def run(self, document: str, question: str) -> BaselineResult:
        """Run the vanilla baseline on a single (document, question) pair.

        Parameters
        ----------
        document:
            The full document text.
        question:
            The question to answer from the document.

        Returns
        -------
        BaselineResult
            Result with ``method="vanilla"``.
        """
        truncated = False
        if len(document) > self.max_chars:
            document = document[: self.max_chars]
            document += "\n\n[... document truncated ...]"
            truncated = True

        prompt = (
            f"Answer the following question based on the provided document.\n\n"
            f"Document:\n{document}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        start = time.perf_counter()
        answer = self.client.completion(prompt)
        elapsed = time.perf_counter() - start

        # Estimate tokens from last-call usage when available; fall back to
        # a rough char/4 heuristic if the client does not expose per-call stats.
        try:
            usage = self.client.get_last_usage()
            total_tokens = usage.total_input_tokens + usage.total_output_tokens
        except Exception:
            total_tokens = (len(prompt) + len(answer)) // 4

        return BaselineResult(
            method="vanilla",
            answer=answer.strip(),
            total_tokens=total_tokens,
            llm_calls=1,
            wall_clock_seconds=elapsed,
            truncated=truncated,
        )
