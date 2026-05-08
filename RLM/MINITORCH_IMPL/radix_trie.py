"""
radix_trie.py — RadixAttention cache for MiniTorch transformers.

Organises KV blocks in a radix trie (prefix tree) keyed by token-ID
sequences.  When multiple requests share a common prefix (e.g. the same
system-prompt tokens), the prefix KV vectors are computed ONCE and
re-used for every subsequent request, avoiding redundant projection work.

Relation to vLLM's RadixAttention
──────────────────────────────────
vLLM's RadixAttention (Zheng et al., 2023) organises PagedAttention
KV blocks in a radix trie indexed by a hash of the token sequence.
This implementation follows the same conceptual design but operates
at single-token granularity (no paging) for clarity on a toy model.

Trie structure
──────────────
  root (no KV)
   ├─ [tok=12] → Node  ← kv_data[layer] = (K_np, V_np) for position 0
   │     ├─ [tok=7]  → Node  ← position 1
   │     │     └─ [tok=3]  → Node  ← position 2  (shared prefix len 3)
   │     └─ [tok=99] → Node  ← different branch at position 1
   └─ [tok=5]  → Node  ← another root child

Each non-root RadixNode stores one token's K,V vectors across ALL
transformer layers as a list of (K_np, V_np) tuples.

Reference counting & LRU eviction
──────────────────────────────────
Active requests increment ref_count on the nodes they are using.  The
eviction policy removes leaf nodes with ref_count == 0 (not in active
use), preferring the least-recently-accessed ones (LRU).
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple
import numpy as np


# Each node's kv_data is:  List[Tuple[K_np, V_np]]  — one entry per layer.
# K_np and V_np have shape  (batch, n_heads, 1, head_dim).
_LayerKV = Tuple[np.ndarray, np.ndarray]


class RadixNode:
    """Single node in the radix trie."""

    __slots__ = (
        "token_id",
        "depth",
        "kv_data",
        "children",
        "ref_count",
        "last_access",
    )

    def __init__(self, token_id: int = -1, depth: int = 0) -> None:
        self.token_id   : int                         = token_id
        self.depth      : int                         = depth
        self.kv_data    : Optional[List[_LayerKV]]    = None
        self.children   : Dict[int, "RadixNode"]      = {}
        self.ref_count  : int                         = 0
        self.last_access: float                       = time.monotonic()

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def touch(self) -> None:
        self.last_access = time.monotonic()


class RadixAttentionCache:
    """
    Radix-trie KV cache enabling prefix sharing across requests.

    Usage pattern
    -------------
    1. Before processing a new request:
         prefix_len, kv_list = cache.lookup_prefix(token_ids)
       Restore kv_list into a KVCacheBuffer if prefix_len > 0.

    2. After processing the full sequence:
         cache.insert(token_ids, per_position_kv_list)
       so future requests can reuse this prefix.

    3. When the request finishes:
         cache.release(token_ids[:prefix_len])
       to decrement ref counts.

    Args:
        n_layers          : Number of transformer layers.
        max_cached_tokens : Maximum total token positions to keep in trie
                            before LRU eviction is triggered.
    """

    def __init__(
        self,
        n_layers: int,
        max_cached_tokens: int = 4096,
    ) -> None:
        self.n_layers          = n_layers
        self.max_cached_tokens = max_cached_tokens
        self.root              = RadixNode(token_id=-1, depth=0)
        self._total_tokens     = 0
        self._queries          = 0   # total tokens queried via lookup_prefix
        self._hits             = 0   # tokens served from cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup_prefix(
        self,
        token_ids: List[int],
    ) -> Tuple[int, List[List[_LayerKV]]]:
        """
        Walk the trie to find the longest matching prefix.

        Args:
            token_ids : Full token sequence for the incoming request.

        Returns:
            prefix_len : Number of tokens found in cache (from the start).
            kv_list    : kv_list[position][layer] = (K_np, V_np).
                         Length equals prefix_len.  The caller should load
                         these into a KVCacheBuffer without recomputing.
        """
        self._queries += len(token_ids)
        node       = self.root
        kv_list    : List[List[_LayerKV]] = []
        prefix_len = 0

        for tok in token_ids:
            if tok not in node.children:
                break
            child = node.children[tok]
            if child.kv_data is None:
                break
            child.touch()
            child.ref_count += 1
            kv_list.append(child.kv_data)
            node       = child
            prefix_len += 1

        self._hits += prefix_len
        return prefix_len, kv_list

    def insert(
        self,
        token_ids: List[int],
        kv_list: List[List[_LayerKV]],
    ) -> None:
        """
        Store KV blocks for all positions of a processed sequence.

        Args:
            token_ids : Token IDs of the completed sequence.
            kv_list   : kv_list[position][layer] = (K_np, V_np).
        """
        if len(token_ids) != len(kv_list):
            raise ValueError(
                f"token_ids length ({len(token_ids)}) != "
                f"kv_list length ({len(kv_list)})"
            )
        node = self.root
        for pos, tok in enumerate(token_ids):
            if tok not in node.children:
                self._maybe_evict()
                child = RadixNode(token_id=tok, depth=node.depth + 1)
                node.children[tok] = child
                self._total_tokens += 1
            child = node.children[tok]
            if child.kv_data is None:
                child.kv_data = kv_list[pos]
            child.touch()
            node = child

    def release(self, token_ids: List[int]) -> None:
        """
        Decrement ref counts along the path for token_ids.
        Call this when a request that used a cached prefix finishes.
        """
        node = self.root
        for tok in token_ids:
            if tok not in node.children:
                break
            child = node.children[tok]
            child.ref_count = max(0, child.ref_count - 1)
            node = child

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def hit_rate(self) -> float:
        """Fraction of queried tokens served from cache (0.0 – 1.0)."""
        return self._hits / max(1, self._queries)

    def stats(self) -> dict:
        return {
            "total_cached_tokens": self._total_tokens,
            "total_queries":       self._queries,
            "total_hits":          self._hits,
            "hit_rate_pct":        round(self.hit_rate() * 100, 2),
        }

    # ------------------------------------------------------------------
    # LRU eviction
    # ------------------------------------------------------------------

    def _maybe_evict(self) -> None:
        """Evict LRU leaf nodes with ref_count == 0 when at capacity."""
        if self._total_tokens < self.max_cached_tokens:
            return
        leaves: List[RadixNode] = []
        self._collect_evictable_leaves(self.root, leaves)
        if not leaves:
            return
        leaves.sort(key=lambda n: n.last_access)
        evict_n = max(1, self.max_cached_tokens // 10)
        for node in leaves[:evict_n]:
            self._remove_leaf(self.root, node)
            node.kv_data = None
            self._total_tokens -= 1

    def _collect_evictable_leaves(
        self, node: RadixNode, result: List[RadixNode]
    ) -> None:
        for child in node.children.values():
            self._collect_evictable_leaves(child, result)
        if node is not self.root and node.is_leaf() and node.ref_count == 0:
            result.append(node)

    def _remove_leaf(self, current: RadixNode, target: RadixNode) -> bool:
        for tok, child in list(current.children.items()):
            if child is target:
                del current.children[tok]
                return True
            if self._remove_leaf(child, target):
                return True
        return False
