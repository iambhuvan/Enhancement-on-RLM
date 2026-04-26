"""
ReAct Baseline
==============
Iterative Thought → Action → Observation loop.

Supported actions
-----------------
- SEARCH: <query>          — keyword search, returns up to 5 passage windows
- READ: <start>-<end>      — return document[start:end] (capped at max_read_chars)
- ANSWER: <text>           — submit final answer and exit the loop

The model runs for at most ``max_iterations`` steps.  If no ANSWER action is
produced within the limit a forced-answer call is made using the accumulated
conversation history.
"""

from __future__ import annotations

import re
import sys
import os
import time
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'BASELINE'))

from rlm.clients.base_lm import BaseLM
from .vanilla import BaselineResult


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You will answer questions about a long document using the following actions:\n"
    "- SEARCH: <query> → returns top 5 matching passages\n"
    "- READ: <start_char>-<end_char> → returns that character range (max 5000 chars)\n"
    "- ANSWER: <your answer> → submits final answer\n"
    "Think step by step before each action."
)

_ACTION_PATTERN = re.compile(r"(SEARCH|READ|ANSWER):\s*(.+)", re.IGNORECASE)

_READ_RANGE_PATTERN = re.compile(r"(\d+)\s*[-–]\s*(\d+)")

# How many characters around a keyword match to include in a SEARCH window
_SEARCH_WINDOW = 300


# ---------------------------------------------------------------------------
# ReActBaseline
# ---------------------------------------------------------------------------


class ReActBaseline:
    """ReAct-style iterative QA baseline.

    Parameters
    ----------
    client:
        Any :class:`rlm.clients.base_lm.BaseLM` instance.
    document:
        Optional document to pre-load.  Can also be supplied (or overridden)
        when calling :meth:`run`.
    max_iterations:
        Hard cap on Thought→Action→Observation cycles.  Default: 10.
    max_read_chars:
        Maximum characters returned by a READ action.  Default: 5 000.
    """

    def __init__(
        self,
        client: BaseLM,
        document: str = "",
        max_iterations: int = 10,
        max_read_chars: int = 5_000,
    ) -> None:
        self.client = client
        self.document = document
        self.max_iterations = max_iterations
        self.max_read_chars = max_read_chars

    # ------------------------------------------------------------------
    # Document tool implementations
    # ------------------------------------------------------------------

    def _search(self, query: str, document: str) -> str:
        """Return up to 5 passage windows that contain any query word."""
        query_words = re.findall(r"\w+", query.lower())
        if not query_words:
            return "[SEARCH] No query terms provided."

        hits: list[tuple[int, str]] = []  # (position, snippet)
        seen_positions: set[int] = set()

        for word in query_words:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            for match in pattern.finditer(document):
                pos = match.start()
                # Deduplicate hits that are within 100 chars of each other
                if any(abs(pos - s) < 100 for s in seen_positions):
                    continue
                seen_positions.add(pos)
                start = max(0, pos - _SEARCH_WINDOW // 2)
                end = min(len(document), pos + _SEARCH_WINDOW // 2)
                snippet = document[start:end]
                hits.append((pos, snippet))
                if len(hits) >= 5:
                    break
            if len(hits) >= 5:
                break

        if not hits:
            return f"[SEARCH] No passages found for query: '{query}'"

        parts = [
            f"[Passage {i + 1} @ char {pos}]:\n...{snippet}..."
            for i, (pos, snippet) in enumerate(hits)
        ]
        return "\n\n".join(parts)

    def _read(self, range_str: str, document: str) -> str:
        """Return a character range from the document."""
        match = _READ_RANGE_PATTERN.search(range_str)
        if not match:
            return f"[READ] Invalid range format '{range_str}'. Expected: <start>-<end>"

        start = int(match.group(1))
        end = int(match.group(2))

        # Clamp to document length
        start = max(0, min(start, len(document)))
        end = max(start, min(end, len(document)))

        # Enforce max_read_chars
        if end - start > self.max_read_chars:
            end = start + self.max_read_chars

        if start == end:
            return "[READ] Empty range or out of document bounds."

        return document[start:end]

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def _dispatch_action(
        self, action: str, argument: str, document: str
    ) -> tuple[str, str | None]:
        """Execute *action* with *argument*.

        Returns
        -------
        (observation, final_answer)
            ``final_answer`` is non-None only when action == 'ANSWER'.
        """
        action_upper = action.upper()

        if action_upper == "SEARCH":
            return self._search(argument.strip(), document), None

        if action_upper == "READ":
            return self._read(argument.strip(), document), None

        if action_upper == "ANSWER":
            return "", argument.strip()

        return f"[Unknown action '{action}']", None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, document: str, question: str) -> BaselineResult:
        """Run the ReAct loop on a (document, question) pair.

        Parameters
        ----------
        document:
            The full document text.
        question:
            The question to answer.

        Returns
        -------
        BaselineResult
            Result with ``method="react"``.
        """
        # Override instance-level document if one is passed
        active_doc = document if document else self.document

        start_time = time.perf_counter()
        llm_calls = 0
        total_tokens = 0

        # Build conversation history (OpenAI-style messages)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Here is the document (may be long — use SEARCH or READ to navigate it):\n\n"
                    f"{active_doc[:2000]}{'...[document continues, use SEARCH/READ to explore]' if len(active_doc) > 2000 else ''}\n\n"
                    f"Question: {question}\n\n"
                    f"Begin. Think step by step, then issue an action."
                ),
            },
        ]

        final_answer: str | None = None

        for iteration in range(self.max_iterations):
            # Call the model
            response = self.client.completion(messages)
            llm_calls += 1

            # Accumulate token usage
            try:
                usage = self.client.get_last_usage()
                total_tokens += usage.total_input_tokens + usage.total_output_tokens
            except Exception:
                total_tokens += len(str(messages)) // 4 + len(response) // 4

            # Append assistant turn
            messages.append({"role": "assistant", "content": response})

            # Parse action from response
            action_match = _ACTION_PATTERN.search(response)

            if not action_match:
                # No recognizable action — prompt for clarification
                messages.append({
                    "role": "user",
                    "content": (
                        "Please use one of the allowed actions: "
                        "SEARCH: <query>, READ: <start>-<end>, or ANSWER: <text>"
                    ),
                })
                continue

            action = action_match.group(1)
            argument = action_match.group(2).strip()

            observation, extracted_answer = self._dispatch_action(action, argument, active_doc)

            if extracted_answer is not None:
                final_answer = extracted_answer
                break

            # Append observation as user turn
            messages.append({
                "role": "user",
                "content": f"[Observation]:\n{observation}\n\nContinue.",
            })

        # If no ANSWER was produced within max_iterations, force a final answer
        if final_answer is None:
            force_prompt = (
                "Based on everything you have gathered so far, "
                "provide your best answer now. "
                "Respond with: ANSWER: <your answer>"
            )
            messages.append({"role": "user", "content": force_prompt})
            force_response = self.client.completion(messages)
            llm_calls += 1

            try:
                usage = self.client.get_last_usage()
                total_tokens += usage.total_input_tokens + usage.total_output_tokens
            except Exception:
                total_tokens += (len(force_prompt) + len(force_response)) // 4

            forced_match = _ACTION_PATTERN.search(force_response)
            if forced_match and forced_match.group(1).upper() == "ANSWER":
                final_answer = forced_match.group(2).strip()
            else:
                # Last resort: use the raw response text
                final_answer = force_response.strip()

        elapsed = time.perf_counter() - start_time

        return BaselineResult(
            method="react",
            answer=final_answer or "",
            total_tokens=total_tokens,
            llm_calls=llm_calls,
            wall_clock_seconds=elapsed,
            truncated=False,
        )
