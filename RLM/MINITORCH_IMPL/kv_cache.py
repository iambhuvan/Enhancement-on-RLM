"""
kv_cache.py — KV Cache buffer for MiniTorch autoregressive generation.

Without KV cache: at decode step t the model re-projects K,V for ALL t tokens
→ O(t) projections + O(t²) attention per step → O(n³) total for n steps.

With KV cache:  at step t the model projects K,V for ONE new token, reads the
rest from this buffer → O(1) projection + O(t) attention → O(n²) total.

The buffer stores K and V as pre-allocated numpy float32 arrays.  There is no
MiniTorch dependency here; callers convert to/from MiniTorch tensors.

Layout
------
  k_cache[layer, batch, head, pos, head_dim]
  v_cache[layer, batch, head, pos, head_dim]

Reference counts on the RadixAttentionCache side are separate; this class
only owns the raw storage for a single in-flight request.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple


class KVCacheBuffer:
    """
    Pre-allocated KV cache for one generation request.

    Args:
        n_layers    : Number of transformer layers.
        n_heads     : Number of attention heads.
        head_dim    : Dimension per head (= n_embd // n_heads).
        max_seq_len : Maximum sequence length to pre-allocate for.
        batch_size  : Batch size (default 1 for auto-regressive generation).
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        max_seq_len: int,
        batch_size: int = 1,
    ) -> None:
        self.n_layers    = n_layers
        self.n_heads     = n_heads
        self.head_dim    = head_dim
        self.max_seq_len = max_seq_len
        self.batch_size  = batch_size
        self.current_len = 0  # tokens currently committed to cache

        # Single contiguous allocation — avoids Python-level per-layer lists.
        # Shape: (n_layers, batch, n_heads, max_seq_len, head_dim)
        self._k = np.zeros(
            (n_layers, batch_size, n_heads, max_seq_len, head_dim),
            dtype=np.float32,
        )
        self._v = np.zeros(
            (n_layers, batch_size, n_heads, max_seq_len, head_dim),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Core write/read API
    # ------------------------------------------------------------------

    def update_and_get(
        self,
        layer_idx: int,
        k_new: np.ndarray,
        v_new: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Write new K,V vectors for `layer_idx` at the current tail of the
        cache, then return the full K,V slice [0 .. current_len + n_new).

        Callers MUST call advance(n_new) after all layers have processed
        the same token(s); otherwise subsequent layers would overwrite the
        wrong positions.

        Args:
            layer_idx : Transformer layer index (0-based).
            k_new     : shape (batch, n_heads, n_new_tokens, head_dim)
            v_new     : shape (batch, n_heads, n_new_tokens, head_dim)

        Returns:
            k_full : (batch, n_heads, current_len + n_new_tokens, head_dim)
            v_full : same shape — both include the newly written tokens.
        """
        n_new = k_new.shape[2]
        end   = self.current_len + n_new
        if end > self.max_seq_len:
            raise RuntimeError(
                f"KVCacheBuffer overflow: tried to write pos {end-1} but "
                f"max_seq_len={self.max_seq_len}"
            )
        self._k[layer_idx, :, :, self.current_len : end, :] = k_new
        self._v[layer_idx, :, :, self.current_len : end, :] = v_new
        # Return copies so callers cannot accidentally mutate the buffer.
        return (
            self._k[layer_idx, :, :, :end, :].copy(),
            self._v[layer_idx, :, :, :end, :].copy(),
        )

    def advance(self, n_new: int = 1) -> None:
        """
        Advance the cache pointer by n_new after all layers have stored
        their K,V for the same token(s).
        """
        self.current_len += n_new

    def reset(self) -> None:
        """Clear cache state so the buffer can be reused for a new request."""
        self.current_len = 0

    # ------------------------------------------------------------------
    # Read-only access (for RadixAttentionCache snapshot / restore)
    # ------------------------------------------------------------------

    def get_layer_kv(self, layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return a copy of cached K,V for `layer_idx` up to current_len."""
        k = self._k[layer_idx, :, :, : self.current_len, :].copy()
        v = self._v[layer_idx, :, :, : self.current_len, :].copy()
        return k, v

    def restore_layer_kv(
        self,
        layer_idx: int,
        k_np: np.ndarray,
        v_np: np.ndarray,
        seq_len: int,
    ) -> None:
        """
        Restore cached K,V from a RadixAttentionCache snapshot.
        Used to seed the buffer with a shared prefix without recomputation.
        """
        self._k[layer_idx, :, :, :seq_len, :] = k_np
        self._v[layer_idx, :, :, :seq_len, :] = v_np

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def memory_bytes(self) -> int:
        """Bytes occupied by currently-cached K,V tensors."""
        per_pos = self.n_heads * self.head_dim * self.batch_size * 4  # float32
        return 2 * self.n_layers * self.current_len * per_pos

    def utilization(self) -> float:
        """Fraction of pre-allocated capacity in use."""
        return self.current_len / self.max_seq_len

    def __repr__(self) -> str:
        return (
            f"KVCacheBuffer("
            f"layers={self.n_layers}, heads={self.n_heads}, "
            f"head_dim={self.head_dim}, "
            f"seq={self.current_len}/{self.max_seq_len}, "
            f"mem={self.memory_bytes()/1024:.1f} KB)"
        )
