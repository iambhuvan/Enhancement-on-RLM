"""
cuda_radix_attention.py — RadixAttention Cache backed by PyTorch Tensors
=========================================================================
Port of radix_trie.py.  The trie structure stays in Python; the KV
tensors that are stored at each leaf are torch.Tensors (CPU-pinned for
fast H2D transfer, or directly on device).

Public interface is identical to RadixAttentionCache so the existing
eval / benchmark code works unchanged.

Algorithm
---------
- Radix trie keyed on token-ID sequences (tuples of int).
- Each fully-inserted prefix stores a list of per-layer (K, V) tensors.
- LRU eviction frees leaf nodes when max_cached_tokens is exceeded.
- Reference counting prevents eviction of in-use entries.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

import torch


class _TrieNode:
    __slots__ = ("children", "kv_list", "token_len", "ref_count", "last_access")

    def __init__(self) -> None:
        self.children: dict[tuple, "_TrieNode"] = {}
        # kv_list[pos] = list of (k_layer0, v_layer0, k_layer1, v_layer1, ...)
        # Stored as list[list[tuple[Tensor, Tensor]]] indexed [pos][layer]
        self.kv_list: list[list[tuple[torch.Tensor, torch.Tensor]]] | None = None
        self.token_len: int = 0          # how many tokens this node represents
        self.ref_count: int = 0
        self.last_access: float = 0.0


class CUDARadixAttentionCache:
    """
    Radix-trie prefix cache whose KV tensors live on CPU pinned memory
    (ready for fast copy to GPU) or directly on the specified device.

    Parameters
    ----------
    n_layers           : number of transformer layers
    max_cached_tokens  : soft cap on total cached tokens before LRU eviction
    pin_memory         : if True, store tensors in CPU pinned memory for fast H2D
    """

    def __init__(
        self,
        n_layers: int,
        max_cached_tokens: int = 4096,
        pin_memory: bool = False,
    ) -> None:
        self.n_layers           = n_layers
        self.max_cached_tokens  = max_cached_tokens
        self.pin_memory         = pin_memory

        self._root = _TrieNode()
        # LRU order: key = token-tuple, value = _TrieNode
        self._lru: OrderedDict[tuple, _TrieNode] = OrderedDict()
        self._cached_tokens: int = 0

        # Stats
        self._hits   = 0
        self._misses = 0
        self._evictions = 0

    # ------------------------------------------------------------------ #
    # Public API (same as RadixAttentionCache)                            #
    # ------------------------------------------------------------------ #

    def lookup_prefix(
        self, prompt_ids: list[int]
    ) -> tuple[int, list[list[tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Walk the trie and return the longest cached prefix.

        Returns
        -------
        (prefix_len, kv_by_position)
            prefix_len      : number of tokens matched
            kv_by_position  : list of length prefix_len;
                              kv_by_position[pos] = list of (k, v) per layer
        """
        node   = self._root
        key    = tuple(prompt_ids)
        length = 0
        result : list[list[tuple[torch.Tensor, torch.Tensor]]] = []

        for i in range(1, len(prompt_ids) + 1):
            sub_key = key[:i]
            if sub_key not in node.children:
                break
            child = node.children[sub_key]
            if child.kv_list is not None:
                length = i
                result = child.kv_list
                child.last_access = time.monotonic()
                # bump in LRU
                if sub_key in self._lru:
                    self._lru.move_to_end(sub_key)
                child.ref_count += 1
            node = child

        if length > 0:
            self._hits += length
        else:
            self._misses += len(prompt_ids)

        return length, result

    def insert(
        self,
        prompt_ids: list[int],
        kv_by_position: list[list[tuple[torch.Tensor, torch.Tensor]]],
    ) -> None:
        """
        Store the per-position KV tensors for a complete prompt in the trie.

        kv_by_position[pos][layer] = (k_tensor, v_tensor)
        Each tensor should be on CPU (optionally pinned).

        Crucially, KV is stored at EVERY node along the path (not just the leaf)
        so that any sub-prefix can be reused across requests that share only a
        common prefix.  Each node at depth i stores kv_by_position[:i], which is
        the full accumulated KV up to that prefix length.  Eviction accounting
        uses token_len = 1 per node so that evicting a node frees exactly one
        token's worth of budget.
        """
        self._maybe_evict(len(prompt_ids))

        node = self._root
        key  = tuple(prompt_ids)

        for i in range(1, len(prompt_ids) + 1):
            sub_key = key[:i]
            if sub_key not in node.children:
                child = _TrieNode()
                node.children[sub_key] = child
            else:
                child = node.children[sub_key]

            # Store accumulated prefix KV at every node (enables prefix sharing)
            if child.kv_list is None:
                child.kv_list     = self._maybe_pin(kv_by_position[:i])
                child.token_len   = 1          # each node = 1 token in the budget
                child.last_access = time.monotonic()
                self._lru[sub_key] = child
                self._cached_tokens += 1

            node = child

    def release(self, prompt_ids: list[int]) -> None:
        """Decrement ref count for every node along the prefix path."""
        key = tuple(prompt_ids)
        for i in range(1, len(prompt_ids) + 1):
            sub_key = key[:i]
            if sub_key in self._lru:
                node = self._lru[sub_key]
                node.ref_count = max(0, node.ref_count - 1)

    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "hits":             self._hits,
            "misses":           self._misses,
            "hit_rate_pct":     (self._hits / total * 100) if total else 0.0,
            "cached_tokens":    self._cached_tokens,
            "max_cached_tokens":self.max_cached_tokens,
            "evictions":        self._evictions,
            "lru_entries":      len(self._lru),
        }

    def clear(self) -> None:
        self._root           = _TrieNode()
        self._lru.clear()
        self._cached_tokens  = 0

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _maybe_pin(
        self,
        kv_by_position: list[list[tuple[torch.Tensor, torch.Tensor]]],
    ) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
        """Optionally move tensors to pinned CPU memory for fast H2D copies."""
        if not self.pin_memory:
            return kv_by_position
        pinned = []
        for pos_kv in kv_by_position:
            layer_list = []
            for k, v in pos_kv:
                pk = k.pin_memory() if k.device.type == "cpu" else k
                pv = v.pin_memory() if v.device.type == "cpu" else v
                layer_list.append((pk, pv))
            pinned.append(layer_list)
        return pinned

    def _maybe_evict(self, incoming_tokens: int) -> None:
        """LRU-evict leaf nodes until there is room for incoming_tokens."""
        while (
            self._cached_tokens + incoming_tokens > self.max_cached_tokens
            and self._lru
        ):
            # Pop least-recently-used entry
            key, node = next(iter(self._lru.items()))
            if node.ref_count > 0:
                # In use — skip and try next
                self._lru.move_to_end(key)
                break
            self._lru.popitem(last=False)
            self._cached_tokens -= node.token_len
            node.kv_list = None
            self._evictions += 1
