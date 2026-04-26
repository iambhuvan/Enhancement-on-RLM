"""
EnhancedRLM (ERLM)
==================
Subclass of the BASELINE ``RLM`` with optional per-optimisation flags.

Optimisations wired here
------------------------
O1 — TF-IDF Prompt Indexer        (enable_indexing)
O2 — Adaptive Budget Controller   (enable_budget)
O3 — Async Subcall Manager        (enable_async)      → injects batched-API system prompt
O4 — KV-cache Prefix Sharing      (enable_kv_cache)   → swaps backend to VLLMPrefixCachedClient
O5 — FP-8 / INT8 Quantisation     (enable_fp8)        → configures quantization on vLLM client
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import Any

# ---- resolve BASELINE package before any rlm imports ----
_BASELINE_PATH = os.path.join(os.path.dirname(__file__), "..", "BASELINE")
_ERLM_PATH = os.path.dirname(__file__)

if _BASELINE_PATH not in sys.path:
    sys.path.insert(0, os.path.abspath(_BASELINE_PATH))
if _ERLM_PATH not in sys.path:
    sys.path.insert(0, os.path.abspath(_ERLM_PATH))

# ---- BASELINE imports (must come after sys.path setup) ----
from rlm.core.rlm import RLM  # noqa: E402
from rlm.core.types import (  # noqa: E402
    ClientBackend,
    EnvironmentType,
    RLMChatCompletion,
    RLMIteration,
)
from rlm.core.lm_handler import LMHandler  # noqa: E402
from rlm.environments.base_env import BaseEnv  # noqa: E402
from rlm.logger import RLMLogger  # noqa: E402

# ---- ERLM optimisation imports ----
from optimisations.prompt_indexer import PromptIndexer  # noqa: E402
from optimisations.budget_controller import AdaptiveBudgetController  # noqa: E402
from optimisations.async_subcall import AsyncSubcallManager  # noqa: E402
from optimisations.kv_prefix_cache import VLLMPrefixCachedClient, create_vllm_client  # noqa: E402
from optimisations.fp8_quantization import recommend_quantization, get_gpu_info  # noqa: E402


class EnhancedRLM(RLM):
    """Enhanced Recursive Language Model with optional system-level optimisations.

    All parameters accepted by the baseline ``RLM`` are forwarded unchanged.
    The additional keyword arguments documented below toggle optimisations that
    are either wired directly in this class (O1, O2) or stored as flags for
    external modules (O3–O5).

    Parameters
    ----------
    enable_indexing:
        O1 — Build a TF-IDF index over the prompt document and inject a
        ``search_context`` tool into the REPL so the model can perform
        sub-linear retrieval instead of linear scanning.
    enable_budget:
        O2 — Track per-iteration productivity and trigger early termination
        when the model is unproductive or the token budget is critical.
    enable_async:
        O3 flag — Stored for use by ``async_subcall.py``.  Has no effect
        inside this class.
    enable_kv_cache:
        O4 flag — Stored for future KV-cache pinning module.
    enable_fp8:
        O5 flag — Stored for future FP-8 quantisation module.
    indexer_chunk_size:
        Character width of each TF-IDF chunk (O1).
    indexer_overlap:
        Overlap between consecutive chunks in characters (O1).
    indexer_top_k:
        Default number of chunks returned by ``search_context`` (O1).
    budget_productivity_threshold:
        Rolling-average productivity below which early termination fires (O2).
    budget_window:
        Number of recent iterations used for the productivity rolling average (O2).
    budget_low_pct:
        Token-fraction remaining that triggers a ``low_budget`` signal (O2).
    budget_critical_pct:
        Token-fraction remaining that triggers a ``critical_budget`` signal (O2).
    """

    def __init__(
        self,
        # ---- baseline params (forwarded verbatim) ----
        backend: ClientBackend = "openai",
        backend_kwargs: dict[str, Any] | None = None,
        environment: EnvironmentType = "local",
        environment_kwargs: dict[str, Any] | None = None,
        depth: int = 0,
        max_depth: int = 1,
        max_iterations: int = 30,
        max_budget: float | None = None,
        max_timeout: float | None = None,
        max_tokens: int | None = None,
        max_errors: int | None = None,
        custom_system_prompt: str | None = None,
        other_backends: list[ClientBackend] | None = None,
        other_backend_kwargs: list[dict[str, Any]] | None = None,
        logger: RLMLogger | None = None,
        verbose: bool = False,
        persistent: bool = False,
        custom_tools: dict[str, Any] | None = None,
        custom_sub_tools: dict[str, Any] | None = None,
        compaction: bool = False,
        compaction_threshold_pct: float = 0.85,
        max_concurrent_subcalls: int = 4,
        on_subcall_start: Callable[[int, str, str], None] | None = None,
        on_subcall_complete: Callable[[int, str, float, str | None], None] | None = None,
        on_iteration_start: Callable[[int, int], None] | None = None,
        on_iteration_complete: Callable[[int, int, float], None] | None = None,
        # ---- ERLM optimisation flags ----
        enable_indexing: bool = False,
        enable_budget: bool = False,
        enable_async: bool = False,
        enable_kv_cache: bool = False,
        enable_fp8: bool = False,
        # ---- O1 indexer params ----
        indexer_chunk_size: int = 2000,
        indexer_overlap: int = 200,
        indexer_top_k: int = 5,
        # ---- O2 budget params ----
        budget_productivity_threshold: float = 0.30,
        budget_window: int = 3,
        budget_low_pct: float = 0.25,
        budget_critical_pct: float = 0.10,
    ) -> None:
        # Store ERLM-specific flags first so methods called during super().__init__
        # can reference them if needed.
        self.enable_indexing = enable_indexing
        self.enable_budget = enable_budget
        self.enable_async = enable_async
        self.enable_kv_cache = enable_kv_cache
        self.enable_fp8 = enable_fp8

        # ---- O1: build indexer, inject tool, augment system prompt ----
        self._indexer: PromptIndexer | None = None

        if enable_indexing:
            self._indexer = PromptIndexer(
                chunk_size=indexer_chunk_size,
                overlap=indexer_overlap,
                top_k=indexer_top_k,
            )
            # Merge search_context into custom_tools
            indexer_tool = self._indexer.get_custom_tool()
            if custom_tools is None:
                custom_tools = {}
            else:
                custom_tools = dict(custom_tools)  # copy to avoid mutating caller's dict
            custom_tools.update(indexer_tool)

            # Prepend guidance note to the system prompt
            _note = (
                "IMPORTANT: Use search_context(query) for fast semantic retrieval "
                "before linear scanning.\n\n"
            )
            if custom_system_prompt:
                custom_system_prompt = _note + custom_system_prompt
            else:
                # Import the baseline prompt so we can prepend the note
                from rlm.utils.prompts import RLM_SYSTEM_PROMPT
                custom_system_prompt = _note + RLM_SYSTEM_PROMPT

        # ---- O2: budget controller ----
        self._budget_controller: AdaptiveBudgetController | None = None

        if enable_budget:
            self._budget_controller = AdaptiveBudgetController(
                productivity_threshold=budget_productivity_threshold,
                window=budget_window,
                low_budget_pct=budget_low_pct,
                critical_budget_pct=budget_critical_pct,
            )

        # ---- O3: async subcall manager — injects batched-API instructions ----
        self._async_manager: AsyncSubcallManager | None = None

        if enable_async:
            self._async_manager = AsyncSubcallManager()
            _async_addon = self._async_manager.get_system_prompt_addon()
            if custom_system_prompt:
                custom_system_prompt = custom_system_prompt + _async_addon
            else:
                from rlm.utils.prompts import RLM_SYSTEM_PROMPT
                custom_system_prompt = RLM_SYSTEM_PROMPT + _async_addon

        # ---- O4 + O5: swap backend to VLLMPrefixCachedClient when kv_cache enabled ----
        # O4 turns on prefix caching; O5 adds quantization.  Both work through vLLM.
        if enable_kv_cache:
            _gpu = get_gpu_info()
            _quant: str | None = None
            if enable_fp8:
                _qcfg = recommend_quantization(_gpu)
                _quant = _qcfg.mode if _qcfg.mode != "none" else None

            # Build the vLLM client and pass it as backend_kwargs so RLM uses it
            _vllm_port = (backend_kwargs or {}).get("port", 8001)
            _vllm_model = (backend_kwargs or {}).get("model", "Qwen/Qwen3-8B")
            _vllm_client = create_vllm_client(
                model_name=_vllm_model,
                quantization=_quant,
                enable_prefix_caching=True,
                port=_vllm_port,
            )
            # Override backend to openai-compatible; inject client into kwargs
            backend = "openai"
            backend_kwargs = backend_kwargs or {}
            backend_kwargs = dict(backend_kwargs)
            backend_kwargs["client"] = _vllm_client

        # ---- delegate to baseline RLM ----
        super().__init__(
            backend=backend,
            backend_kwargs=backend_kwargs,
            environment=environment,
            environment_kwargs=environment_kwargs,
            depth=depth,
            max_depth=max_depth,
            max_iterations=max_iterations,
            max_budget=max_budget,
            max_timeout=max_timeout,
            max_tokens=max_tokens,
            max_errors=max_errors,
            custom_system_prompt=custom_system_prompt,
            other_backends=other_backends,
            other_backend_kwargs=other_backend_kwargs,
            logger=logger,
            verbose=verbose,
            persistent=persistent,
            custom_tools=custom_tools,
            custom_sub_tools=custom_sub_tools,
            compaction=compaction,
            compaction_threshold_pct=compaction_threshold_pct,
            max_concurrent_subcalls=max_concurrent_subcalls,
            on_subcall_start=on_subcall_start,
            on_subcall_complete=on_subcall_complete,
            on_iteration_start=on_iteration_start,
            on_iteration_complete=on_iteration_complete,
        )

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def completion(
        self,
        prompt: str | dict[str, Any],
        root_prompt: str | None = None,
    ) -> RLMChatCompletion:
        """Run the ERLM completion pipeline with optional O1/O2 wrappers.

        If O1 (indexing) is enabled and *prompt* is a plain string, the
        TF-IDF index is built over the prompt before the RLM loop starts.

        If O2 (budget) is enabled, the budget controller is reset so that
        state from a previous call does not leak into this one.

        Parameters
        ----------
        prompt:
            The document / question to process.  May be a plain string or a
            message-dict accepted by the underlying LM client.
        root_prompt:
            Optional short question displayed to the root model alongside the
            full document context (forwarded to baseline).

        Returns
        -------
        RLMChatCompletion
            The standard baseline completion result.
        """
        # O1: build TF-IDF index when a plain-string document is provided
        if self.enable_indexing and self._indexer is not None and isinstance(prompt, str):
            self._indexer.build_index(prompt)

        # O2: reset productivity/token tracking for this call
        if self.enable_budget and self._budget_controller is not None:
            self._budget_controller.reset()

        return super().completion(prompt, root_prompt)

    def _completion_turn(
        self,
        prompt: str | dict[str, Any],
        lm_handler: LMHandler,
        environment: BaseEnv,
    ) -> RLMIteration:
        """Perform one iteration of the RLM loop with O2 budget monitoring.

        The baseline implementation is called first to obtain the
        ``RLMIteration``.  When O2 is active the iteration's response and
        current token count are recorded; if the controller signals early
        termination the ``final_answer`` field is patched so the outer loop
        exits cleanly on this iteration.

        Parameters
        ----------
        prompt:
            Current message history passed to the LM.
        lm_handler:
            Active ``LMHandler`` for this completion call.
        environment:
            Active REPL environment for this completion call.

        Returns
        -------
        RLMIteration
            The iteration object, potentially with ``final_answer`` set to
            trigger early termination.
        """
        iteration: RLMIteration = super()._completion_turn(
            prompt=prompt,
            lm_handler=lm_handler,
            environment=environment,
        )

        if self.enable_budget and self._budget_controller is not None:
            # Obtain cumulative token count from the handler
            usage = lm_handler.get_usage_summary()
            tokens_used = (
                usage.total_input_tokens + usage.total_output_tokens
            )

            self._budget_controller.record_iteration(
                response=iteration.response,
                tokens_used=tokens_used,
            )

            # Only check against max_tokens when a limit is configured
            effective_max = self.max_tokens if self.max_tokens is not None else 0

            if effective_max > 0:
                should_stop, reason = self._budget_controller.should_terminate_early(
                    tokens_used=tokens_used,
                    max_tokens=effective_max,
                )
                if should_stop and iteration.final_answer is None:
                    iteration.final_answer = (
                        self._best_partial_answer
                        or f"Budget terminated early: {reason}"
                    )

        return iteration

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def erlm_config(self) -> dict[str, Any]:
        """Return a snapshot of which ERLM optimisations are active.

        Returns
        -------
        dict
            Keys are optimisation names; values are ``True``/``False``
            depending on whether the feature is enabled for this instance.
        """
        cfg: dict[str, Any] = {
            "enable_indexing": self.enable_indexing,
            "enable_budget": self.enable_budget,
            "enable_async": self.enable_async,
            "enable_kv_cache": self.enable_kv_cache,
            "enable_fp8": self.enable_fp8,
        }
        if self._async_manager is not None:
            cfg["async_speedup_stats"] = self._async_manager.get_speedup_stats()
        return cfg
