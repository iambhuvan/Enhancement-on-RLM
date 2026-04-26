"""
O3: Async Parallel Subcall Manager
====================================
Tracks timing of sequential vs parallel subcall execution and modifies
the RLM system prompt to strongly encourage batched LLM/RLM queries.

Key idea: When the RLM model calls llm_query() in a loop, each call is
serialised (sequential). By instructing the model to use
llm_query_batched([p1, p2, ...]) instead, the LMHandler executes all
prompts concurrently via asyncio.gather — yielding near-linear speedup
proportional to the number of independent subcalls.
"""

import time
from threading import Lock
from typing import Any


# ---------------------------------------------------------------------------
# System-prompt addon text
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_ADDON = """\

===========================================================================
PERFORMANCE CRITICAL — PARALLEL EXECUTION INSTRUCTIONS (READ CAREFULLY)
===========================================================================
When you need to process multiple independent chunks of text, answer
multiple independent questions, or run multiple independent sub-tasks,
you MUST use the batched variants instead of looping over individual calls:

  • llm_query_batched([prompt1, prompt2, prompt3, ...])
      Runs ALL prompts in PARALLEL.  This is dramatically faster than
      calling llm_query() in a loop, which runs each call sequentially.

  • rlm_query_batched([prompt1, prompt2, prompt3, ...])
      Same contract for recursive RLM sub-calls.  Use whenever you would
      otherwise write:
          for p in prompts:
              rlm_query(p)

WHEN TO USE BATCHED CALLS
  - Summarising multiple document chunks independently.
  - Asking the same question across different contexts.
  - Running multiple verification or analysis steps that do not depend on
    each other's outputs.
  - Extracting information from several passages in parallel.

WHEN NOT TO USE BATCHED CALLS
  - Step B depends on the output of Step A (sequential dependency).
  - A single prompt covers all the information you need.

EXAMPLE — WRONG (sequential, slow):
    result_a = llm_query("Summarise chunk A")
    result_b = llm_query("Summarise chunk B")
    result_c = llm_query("Summarise chunk C")

EXAMPLE — CORRECT (parallel, fast):
    results = llm_query_batched([
        "Summarise chunk A",
        "Summarise chunk B",
        "Summarise chunk C",
    ])
    result_a, result_b, result_c = results

Failing to use the batched API when multiple independent queries are
needed is a performance bug.  Always prefer batched variants.
===========================================================================
"""


# ---------------------------------------------------------------------------
# AsyncSubcallManager
# ---------------------------------------------------------------------------


class AsyncSubcallManager:
    """
    Tracks timing statistics for sequential-equivalent vs actual parallel
    subcall execution, and provides a system-prompt addon that instructs
    the RLM model to use batched query APIs.

    Thread-safe: all internal accumulators are protected by a ``threading.Lock``.

    Usage
    -----
    manager = AsyncSubcallManager(max_workers=8)

    # Inject into the RLM system prompt:
    system_prompt = base_system_prompt + manager.get_system_prompt_addon()

    # After a batched call completes, record timings:
    manager.record_sequential_time(n_calls=4, per_call_time=1.2)
    manager.record_parallel_time(n_calls=4, total_time=1.4)

    stats = manager.get_speedup_stats()
    """

    def __init__(self, max_workers: int = 8) -> None:
        """
        Parameters
        ----------
        max_workers:
            Maximum number of concurrent worker threads/coroutines used
            when executing a batched subcall.  Stored for reference; the
            actual concurrency is enforced by the LMHandler's semaphore.
        """
        self.max_workers: int = max_workers

        self._lock = Lock()
        self._sequential_equivalent_time: float = 0.0
        self._actual_parallel_time: float = 0.0
        self._total_parallel_batches: int = 0
        self._total_calls_parallelized: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_system_prompt_addon(self) -> str:
        """
        Return a string to prepend (or append) to the RLM system prompt
        that instructs the model to always prefer batched query APIs.

        Returns
        -------
        str
            Multi-line instruction block.
        """
        return _SYSTEM_PROMPT_ADDON

    def record_sequential_time(self, n_calls: int, per_call_time: float) -> None:
        """
        Record what the total wall-clock time *would have been* if the
        ``n_calls`` subcalls had been executed sequentially.

        Parameters
        ----------
        n_calls:
            Number of individual prompts in the batch.
        per_call_time:
            Estimated or measured wall-clock seconds for a *single* call
            (e.g. the average observed single-call latency).
        """
        sequential_total = n_calls * per_call_time
        with self._lock:
            self._sequential_equivalent_time += sequential_total

    def record_parallel_time(self, n_calls: int, total_time: float) -> None:
        """
        Record the actual wall-clock time taken by a parallel batch.

        Parameters
        ----------
        n_calls:
            Number of prompts that were executed in this batch.
        total_time:
            Wall-clock seconds from batch-start to the last result being
            returned (i.e. ``asyncio.gather`` wall time).
        """
        with self._lock:
            self._actual_parallel_time += total_time
            self._total_parallel_batches += 1
            self._total_calls_parallelized += n_calls

    def get_speedup_stats(self) -> dict[str, Any]:
        """
        Return a dictionary summarising the cumulative speedup achieved
        by parallelising subcalls.

        Returns
        -------
        dict with keys:
            sequential_equivalent_time : float
                Total seconds that would have been spent in sequential mode.
            actual_parallel_time : float
                Total seconds actually spent in parallel mode.
            speedup_ratio : float
                sequential_equivalent_time / actual_parallel_time.
                Returns 0.0 when no parallel time has been recorded yet.
            total_parallel_batches : int
                Number of ``llm_query_batched`` / ``rlm_query_batched``
                calls that have been measured.
            total_calls_parallelized : int
                Aggregate number of individual prompts that ran in parallel.
        """
        with self._lock:
            seq = self._sequential_equivalent_time
            par = self._actual_parallel_time
            speedup = (seq / par) if par > 0.0 else 0.0
            return {
                "sequential_equivalent_time": seq,
                "actual_parallel_time": par,
                "speedup_ratio": speedup,
                "total_parallel_batches": self._total_parallel_batches,
                "total_calls_parallelized": self._total_calls_parallelized,
            }

    def reset(self) -> None:
        """Reset all accumulated timing statistics to zero."""
        with self._lock:
            self._sequential_equivalent_time = 0.0
            self._actual_parallel_time = 0.0
            self._total_parallel_batches = 0
            self._total_calls_parallelized = 0


# ---------------------------------------------------------------------------
# SubcallTimingWrapper
# ---------------------------------------------------------------------------


class SubcallTimingWrapper:
    """
    Wraps any callable so that each invocation is timed and the measured
    latency is recorded in an :class:`AsyncSubcallManager`.

    Designed for wrapping single-call functions (e.g. ``client.completion``)
    so that per-call latencies are captured for sequential-time estimation.

    Example
    -------
    manager = AsyncSubcallManager()
    wrapped_completion = SubcallTimingWrapper(client.completion, manager)

    # Each call records 1 sequential call of measured duration:
    result = wrapped_completion("Summarise this text.")
    """

    def __init__(self, fn: callable, manager: AsyncSubcallManager) -> None:
        """
        Parameters
        ----------
        fn:
            The callable to wrap.  Must be a synchronous callable;
            for async functions wrap with an async shim if needed.
        manager:
            The :class:`AsyncSubcallManager` that will receive timing
            data after each invocation.
        """
        self._fn = fn
        self._manager = manager
        # Preserve the name for debugging / repr
        self.__name__: str = getattr(fn, "__name__", repr(fn))
        self.__doc__: str | None = getattr(fn, "__doc__", None)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Invoke the wrapped callable, record the elapsed wall-clock time
        as one sequential call in the manager, and return the result.

        Returns
        -------
        Any
            Whatever the wrapped callable returns.
        """
        start = time.perf_counter()
        try:
            result = self._fn(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            # Record as n_calls=1 sequential event with measured latency
            self._manager.record_sequential_time(n_calls=1, per_call_time=elapsed)
        return result

    def __repr__(self) -> str:
        return f"SubcallTimingWrapper(fn={self.__name__!r})"
