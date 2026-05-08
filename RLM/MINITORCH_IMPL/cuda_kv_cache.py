"""
cuda_kv_cache.py — PyTorch KV Cache Buffer (CUDA / MPS / CPU)
=============================================================
Port of kv_cache.py from numpy → torch.Tensor.
Runs on any device: CUDA GPU, Apple MPS, or CPU fallback.

Same public interface as KVCacheBuffer so existing code that
calls update_and_get / advance / reset / memory_bytes / utilization
works without changes.
"""
from __future__ import annotations

import torch


def best_device() -> torch.device:
    """Return the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CUDAKVCacheBuffer:
    """
    Pre-allocated KV cache backed by torch.Tensor on GPU/MPS/CPU.

    Layout: _k[layer, head, seq_pos, head_dim]
            _v[layer, head, seq_pos, head_dim]

    Parameters
    ----------
    n_layers    : number of transformer layers
    n_heads     : number of attention heads
    head_dim    : dimension per head (n_embd // n_heads)
    max_seq_len : maximum sequence length that can be cached
    device      : torch device ("cuda", "mps", "cpu", or None for auto)
    dtype       : tensor dtype (default float16 on GPU, float32 on CPU)
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        max_seq_len: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.n_layers    = n_layers
        self.n_heads     = n_heads
        self.head_dim    = head_dim
        self.max_seq_len = max_seq_len

        self.device = torch.device(device) if device else best_device()

        # Use float16 on GPU for memory efficiency; float32 on CPU
        if dtype is not None:
            self.dtype = dtype
        elif self.device.type in ("cuda", "mps"):
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        # Allocate once — shape: (n_layers, n_heads, max_seq_len, head_dim)
        self._k = torch.zeros(
            n_layers, n_heads, max_seq_len, head_dim,
            dtype=self.dtype, device=self.device
        )
        self._v = torch.zeros_like(self._k)

        self.current_len: int = 0   # how many positions are valid

    # ------------------------------------------------------------------ #
    # Core API                                                             #
    # ------------------------------------------------------------------ #

    def update_and_get(
        self,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Write new K,V for one token at position current_len, then return
        the full K,V slice [0 .. current_len] for this layer.

        Parameters
        ----------
        layer_idx : which transformer layer
        k_new     : (n_heads, 1, head_dim) or (n_heads, head_dim)
        v_new     : same shape as k_new

        Returns
        -------
        k_full : (n_heads, current_len+1, head_dim)
        v_full : (n_heads, current_len+1, head_dim)
        """
        pos = self.current_len

        # Normalise to (n_heads, head_dim) for writing
        k_write = k_new.view(self.n_heads, self.head_dim).to(self.dtype)
        v_write = v_new.view(self.n_heads, self.head_dim).to(self.dtype)

        self._k[layer_idx, :, pos, :] = k_write
        self._v[layer_idx, :, pos, :] = v_write

        k_full = self._k[layer_idx, :, : pos + 1, :]   # (H, pos+1, D)
        v_full = self._v[layer_idx, :, : pos + 1, :]
        return k_full, v_full

    def advance(self) -> None:
        """Move the write pointer forward by one after all layers are done."""
        self.current_len += 1

    def reset(self) -> None:
        """Clear the cache (zero fill + reset pointer)."""
        self._k.zero_()
        self._v.zero_()
        self.current_len = 0

    def restore_layer_kv(
        self,
        layer_idx: int,
        k_prefix: torch.Tensor,
        v_prefix: torch.Tensor,
        prefix_len: int,
    ) -> None:
        """
        Write a pre-computed prefix's KV tensors into the cache.
        Used by RadixAttention to restore a cached prefix.

        k_prefix : (n_heads, prefix_len, head_dim)
        v_prefix : same
        """
        self._k[layer_idx, :, :prefix_len, :] = k_prefix.to(dtype=self.dtype, device=self.device)
        self._v[layer_idx, :, :prefix_len, :] = v_prefix.to(dtype=self.dtype, device=self.device)

    def snapshot_layer_kv(
        self, layer_idx: int, length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return a CPU copy of the first *length* positions of layer *layer_idx*.
        Used by RadixAttention.insert() to persist prefix KV.

        Returns
        -------
        k : (n_heads, length, head_dim)  on CPU
        v : same
        """
        k = self._k[layer_idx, :, :length, :].detach().cpu()
        v = self._v[layer_idx, :, :length, :].detach().cpu()
        return k, v

    # ------------------------------------------------------------------ #
    # Memory helpers                                                       #
    # ------------------------------------------------------------------ #

    def memory_bytes(self) -> int:
        """Total bytes allocated for the KV buffers."""
        elem_bytes = self._k.element_size()
        total_elems = 2 * self.n_layers * self.n_heads * self.max_seq_len * self.head_dim
        return total_elems * elem_bytes

    def utilization(self) -> float:
        """Fraction of max_seq_len that has been written."""
        return self.current_len / max(1, self.max_seq_len)

    def __repr__(self) -> str:
        kb = self.memory_bytes() / 1024
        return (
            f"CUDAKVCacheBuffer(layers={self.n_layers}, heads={self.n_heads}, "
            f"head_dim={self.head_dim}, max_seq={self.max_seq_len}, "
            f"device={self.device}, dtype={self.dtype}, "
            f"used={self.current_len}/{self.max_seq_len}, {kb:.1f}KB)"
        )
