"""
O2: Adaptive Budget Controller
==============================
Tracks per-iteration productivity and injects early-termination signals
into the RLM iteration loop when the model is unproductive or the token
budget is running low.
"""

from __future__ import annotations

import re


def _tokenize(text: str) -> set[str]:
    """Split *text* into a set of lowercase word tokens."""
    return set(re.findall(r"\b\w+\b", text.lower()))


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two token sets.

    Returns 0.0 when both sets are empty to avoid zero-division.
    """
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


class AdaptiveBudgetController:
    """Monitor per-iteration productivity and signal early termination.

    Two orthogonal termination axes are tracked:

    1. **Productivity** — Jaccard similarity between consecutive responses is
       used to measure how much *new* information the model is generating.
       When the rolling-window average drops below *productivity_threshold*
       the controller signals termination.

    2. **Budget** — The fraction of *max_tokens* consumed is compared against
       two configurable thresholds: a soft warning (``low_budget``) and a hard
       cut-off (``critical_budget``).

    Parameters
    ----------
    productivity_threshold:
        Minimum acceptable rolling-average productivity (0–1).  Below this
        value the model is considered to be spinning its wheels.
    window:
        Number of recent iterations used for the rolling productivity average.
    low_budget_pct:
        Fraction of the remaining budget that triggers a ``low_budget`` signal.
        For example, 0.25 means "warn when only 25 % of tokens are left".
    critical_budget_pct:
        Fraction of the remaining budget that triggers a ``critical_budget``
        signal.  For example, 0.10 means "stop when only 10 % of tokens are
        left".
    """

    def __init__(
        self,
        productivity_threshold: float = 0.30,
        window: int = 3,
        low_budget_pct: float = 0.25,
        critical_budget_pct: float = 0.10,
    ) -> None:
        if not (0.0 <= productivity_threshold <= 1.0):
            raise ValueError("productivity_threshold must be in [0, 1].")
        if window < 1:
            raise ValueError("window must be at least 1.")
        if not (0.0 < low_budget_pct <= 1.0):
            raise ValueError("low_budget_pct must be in (0, 1].")
        if not (0.0 < critical_budget_pct <= low_budget_pct):
            raise ValueError(
                "critical_budget_pct must be in (0, low_budget_pct]."
            )

        self.productivity_threshold = productivity_threshold
        self.window = window
        self.low_budget_pct = low_budget_pct
        self.critical_budget_pct = critical_budget_pct

        # Mutable state — reset per query via reset()
        self._responses: list[str] = []
        self._tokens_per_iter: list[int] = []
        self.productivity_history: list[float] = []
        self.termination_reason: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def iteration_count(self) -> int:
        """Number of iterations recorded since the last ``reset``."""
        return len(self._responses)

    def record_iteration(self, response: str, tokens_used: int) -> None:
        """Record the model's response and cumulative token count for this iteration.

        Parameters
        ----------
        response:
            The raw text response produced by the model in this iteration.
        tokens_used:
            Cumulative total tokens consumed so far (input + output).
        """
        self._responses.append(response or "")
        self._tokens_per_iter.append(tokens_used)

        # Compute and store productivity for this iteration
        if len(self._responses) >= 2:
            productivity = self.compute_productivity()
        else:
            # First iteration: no prior response to compare against
            productivity = 1.0

        self.productivity_history.append(productivity)

    def compute_productivity(self) -> float:
        """Compute the productivity score for the most recent iteration.

        Productivity is defined as ``1 - jaccard(current, prev)`` which
        represents the fraction of *new* information in the current response
        relative to the previous one.  A score near 1 means the model is
        generating entirely new content; near 0 means it is repeating itself.

        Returns
        -------
        float
            Productivity score in [0, 1].  Returns 1.0 if fewer than two
            responses have been recorded.
        """
        if len(self._responses) < 2:
            return 1.0

        current_tokens = _tokenize(self._responses[-1])
        prev_tokens = _tokenize(self._responses[-2])
        jaccard_sim = _jaccard(current_tokens, prev_tokens)
        return max(0.0, min(1.0, 1.0 - jaccard_sim))

    def should_terminate_early(
        self, tokens_used: int, max_tokens: int
    ) -> tuple[bool, str]:
        """Decide whether the RLM should stop before the next iteration.

        Checks are evaluated in order of severity:

        1. ``critical_budget``: tokens_used / max_tokens > 1 - critical_budget_pct
        2. ``low_budget``:      tokens_used / max_tokens > 1 - low_budget_pct
        3. ``low_productivity``: rolling-window average productivity < threshold

        Parameters
        ----------
        tokens_used:
            Cumulative tokens consumed so far.
        max_tokens:
            The hard token ceiling configured on the RLM.

        Returns
        -------
        tuple[bool, str]
            ``(should_stop, reason)`` where *reason* is one of
            ``"critical_budget"``, ``"low_budget"``, ``"low_productivity"``,
            or ``""`` when no termination is warranted.
        """
        if max_tokens <= 0:
            return False, ""

        budget_fraction = tokens_used / max_tokens

        # Critical budget check (most severe — check first)
        if budget_fraction > (1.0 - self.critical_budget_pct):
            reason = "critical_budget"
            self.termination_reason = reason
            return True, reason

        # Low budget check
        if budget_fraction > (1.0 - self.low_budget_pct):
            reason = "low_budget"
            self.termination_reason = reason
            return True, reason

        # Low productivity check (need at least `window` data points)
        if len(self.productivity_history) >= self.window:
            recent = self.productivity_history[-self.window :]
            rolling_avg = sum(recent) / len(recent)
            if rolling_avg < self.productivity_threshold:
                reason = "low_productivity"
                self.termination_reason = reason
                return True, reason

        return False, ""

    def reset(self) -> None:
        """Reset all state for a new query.

        Call this at the beginning of each ``completion()`` invocation so
        productivity history from a prior query does not bleed into the next.
        """
        self._responses = []
        self._tokens_per_iter = []
        self.productivity_history = []
        self.termination_reason = None
