"""
cuda_decoder_lm.py — GPT-2 style Decoder LM in PyTorch with CUDA KV Cache
===========================================================================
Full transformer decoder using:
  - torch.nn.functional.scaled_dot_product_attention (Flash Attention on GPU)
  - CUDAKVCacheBuffer   for O(n) incremental decoding
  - CUDARadixAttentionCache  for O(1) prefix reuse across requests

Three generation modes (same API as KVDecoderLM):
  generate_no_cache()   — naive O(n³), recomputes full attention each step
  generate_with_cache() — O(n²) with KV cache, one forward pass per new token
  generate_with_radix() — O(n) amortised with RadixAttention prefix sharing

Can optionally load real GPT-2 weights from HuggingFace.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cuda_kv_cache import CUDAKVCacheBuffer, best_device
from cuda_radix_attention import CUDARadixAttentionCache


# ══════════════════════════════════════════════════════════════════════════════
# Building blocks
# ══════════════════════════════════════════════════════════════════════════════

class LayerNorm(nn.Module):
    def __init__(self, n_embd: int, eps: float = 1e-5, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias   = nn.Parameter(torch.zeros(n_embd)) if bias else None
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Supports two forward modes:
      full(x)           — attends over the full sequence x  (no cache)
      incremental(x, kv_cache, layer_idx) — one-token step using KV cache
    """

    def __init__(self, n_embd: int, n_head: int, p_dropout: float = 0.0,
                 bias: bool = True) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head   = n_head
        self.head_dim = n_embd // n_head
        self.scale    = math.sqrt(self.head_dim)

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd,     bias=bias)
        self.drop   = nn.Dropout(p_dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) → (B, H, T, D)"""
        B, T, C = x.shape
        x = x.view(B, T, self.n_head, self.head_dim)
        return x.transpose(1, 2)

    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        """Full-sequence attention for generate_no_cache."""
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)  # each (B, T, C)
        q = self._split_heads(q)   # (B, H, T, D)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Flash Attention on GPU; standard scaled dot-product on CPU
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.drop.p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

    def forward_incremental(
        self,
        x: torch.Tensor,            # (B, 1, C) — single new token
        kv_cache: CUDAKVCacheBuffer,
        layer_idx: int,
    ) -> torch.Tensor:
        """One-step attention using KV cache."""
        B, _, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)  # each (B, 1, C)

        # Reshape to (B*H, 1, D) — we process batch=1 for simplicity
        q = self._split_heads(q)   # (B, H, 1, D)
        k = self._split_heads(k)   # (B, H, 1, D)
        v = self._split_heads(v)

        # Update cache and get full K, V
        k_full, v_full = kv_cache.update_and_get(
            layer_idx,
            k[0],    # (H, 1, D)
            v[0],    # (H, 1, D)
        )
        # k_full : (H, pos+1, D), v_full : (H, pos+1, D)
        # Add batch dimension
        k_full = k_full.unsqueeze(0).to(q.dtype)   # (1, H, pos+1, D)
        v_full = v_full.unsqueeze(0).to(q.dtype)

        # q: (1, H, 1, D)  k_full: (1, H, pos+1, D)  — causal ok (query is last)
        y = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, 1, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, n_embd: int, bias: bool = True) -> None:
        super().__init__()
        self.fc   = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.gelu(self.fc(x)))


class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int,
                 p_dropout: float = 0.0, ln_eps: float = 1e-5,
                 bias: bool = True) -> None:
        super().__init__()
        self.ln1  = LayerNorm(n_embd, ln_eps, bias)
        self.attn = CausalSelfAttention(n_embd, n_head, p_dropout, bias)
        self.ln2  = LayerNorm(n_embd, ln_eps, bias)
        self.mlp  = MLP(n_embd, bias)

    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn.forward_full(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def forward_incremental(
        self, x: torch.Tensor,
        kv_cache: CUDAKVCacheBuffer,
        layer_idx: int,
    ) -> torch.Tensor:
        x = x + self.attn.forward_incremental(self.ln1(x), kv_cache, layer_idx)
        x = x + self.mlp(self.ln2(x))
        return x


# ══════════════════════════════════════════════════════════════════════════════
# Main LM
# ══════════════════════════════════════════════════════════════════════════════

class CUDADecoderLM(nn.Module):
    """
    GPT-2 style decoder LM with CUDA KV cache + RadixAttention.

    Parameters match GPT-2 small by default (set n_vocab=50257, n_embd=768,
    n_head=12, n_layers=12 to load real weights).
    """

    N_LAYERS: int = 4   # default for testing; overridden in __init__

    def __init__(
        self,
        n_vocab:     int   = 256,
        n_embd:      int   = 128,
        n_head:      int   = 4,
        n_layers:    int   = 4,
        n_positions: int   = 512,
        p_dropout:   float = 0.0,
        ln_eps:      float = 1e-5,
        bias:        bool  = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.n_vocab     = n_vocab
        self.n_embd      = n_embd
        self.n_head      = n_head
        self.n_layers    = n_layers
        self.n_positions = n_positions
        self.head_dim    = n_embd // n_head

        CUDADecoderLM.N_LAYERS = n_layers

        self.device = device or best_device()

        self.tok_emb = nn.Embedding(n_vocab, n_embd)
        self.pos_emb = nn.Embedding(n_positions, n_embd)
        self.drop    = nn.Dropout(p_dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, p_dropout, ln_eps, bias)
            for _ in range(n_layers)
        ])

        self.ln_f   = LayerNorm(n_embd, ln_eps, bias)
        self.lm_head = nn.Linear(n_embd, n_vocab, bias=False)

        # Weight tying (GPT-2 style)
        self.tok_emb.weight = self.lm_head.weight

        self.to(self.device)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def make_kv_cache(self, max_seq_len: int) -> CUDAKVCacheBuffer:
        return CUDAKVCacheBuffer(
            n_layers=self.n_layers,
            n_heads=self.n_head,
            head_dim=self.head_dim,
            max_seq_len=max_seq_len,
            device=self.device,
        )

    # ------------------------------------------------------------------ #
    # Forward passes                                                      #
    # ------------------------------------------------------------------ #

    def _forward_full(self, ids: torch.Tensor) -> torch.Tensor:
        """Full-sequence forward: (B, T) → (B, T, vocab)."""
        B, T = ids.shape
        pos  = torch.arange(T, device=self.device)
        x    = self.drop(self.tok_emb(ids) + self.pos_emb(pos))
        for block in self.blocks:
            x = block.forward_full(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def _decode_step(
        self,
        token_id: int,
        abs_pos:  int,
        kv_cache: CUDAKVCacheBuffer,
    ) -> torch.Tensor:
        """
        One incremental decode step.
        Returns logits: (1, 1, vocab).
        """
        ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        pos = torch.tensor([abs_pos],   dtype=torch.long, device=self.device)
        x   = self.tok_emb(ids) + self.pos_emb(pos)   # (1, 1, n_embd)

        for i, block in enumerate(self.blocks):
            x = block.forward_incremental(x, kv_cache, i)

        kv_cache.advance()
        x = self.ln_f(x)
        return self.lm_head(x)   # (1, 1, vocab)

    # ------------------------------------------------------------------ #
    # Generation methods                                                  #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate_no_cache(
        self, prompt_ids: list[int], max_new_tokens: int
    ) -> list[int]:
        """
        Naive O(n³): full forward on growing sequence each step.
        Identical output to generate_with_cache (correctness reference).
        """
        self.eval()
        tokens = list(prompt_ids)
        for _ in range(max_new_tokens):
            ids    = torch.tensor([tokens], dtype=torch.long, device=self.device)
            logits = self._forward_full(ids)          # (1, T, vocab)
            next_t = int(logits[0, -1, :].argmax())
            tokens.append(next_t)
        return tokens

    @torch.no_grad()
    def generate_with_cache(
        self, prompt_ids: list[int], max_new_tokens: int
    ) -> list[int]:
        """
        O(n²): KV cache, one decode step per new token.
        """
        self.eval()
        total    = len(prompt_ids) + max_new_tokens + 1
        kv_cache = self.make_kv_cache(total)

        # Prefill: process prompt tokens incrementally to fill cache
        last_logits = None
        for pos, tok in enumerate(prompt_ids):
            last_logits = self._decode_step(tok, pos, kv_cache)

        next_tok = int(last_logits[0, 0, :].argmax())
        tokens   = list(prompt_ids) + [next_tok]

        for step in range(1, max_new_tokens):
            pos        = len(prompt_ids) + step - 1
            last_logits = self._decode_step(next_tok, pos, kv_cache)
            next_tok    = int(last_logits[0, 0, :].argmax())
            tokens.append(next_tok)

        return tokens

    @torch.no_grad()
    def generate_with_radix(
        self,
        prompt_ids:     list[int],
        max_new_tokens: int,
        radix_cache:    CUDARadixAttentionCache,
    ) -> list[int]:
        """
        O(n) amortised: restore cached prefix, decode only new tokens.
        """
        self.eval()
        total    = len(prompt_ids) + max_new_tokens + 1
        kv_cache = self.make_kv_cache(total)

        # ── 1. Look up prefix in radix cache ──────────────────────────────
        prefix_len, kv_by_pos = radix_cache.lookup_prefix(prompt_ids)

        if prefix_len > 0:
            # Restore prefix KV into the GPU cache
            for layer_idx in range(self.n_layers):
                k_parts = [kv_by_pos[p][layer_idx][0] for p in range(prefix_len)]
                v_parts = [kv_by_pos[p][layer_idx][1] for p in range(prefix_len)]
                k_prefix = torch.cat(k_parts, dim=1)   # (H, prefix_len, D)
                v_prefix = torch.cat(v_parts, dim=1)
                kv_cache.restore_layer_kv(layer_idx, k_prefix, v_prefix, prefix_len)
            kv_cache.current_len = prefix_len - 1

        # ── 2. Decode from the first uncached token ───────────────────────
        decode_start = max(0, prefix_len - 1)
        last_logits  = None

        for abs_pos in range(decode_start, len(prompt_ids)):
            last_logits = self._decode_step(prompt_ids[abs_pos], abs_pos, kv_cache)

        next_tok = int(last_logits[0, 0, :].argmax())
        tokens   = list(prompt_ids) + [next_tok]

        for step in range(1, max_new_tokens):
            pos         = len(prompt_ids) + step - 1
            last_logits = self._decode_step(next_tok, pos, kv_cache)
            next_tok    = int(last_logits[0, 0, :].argmax())
            tokens.append(next_tok)

        # ── 3. Snapshot new KV and insert into radix cache ────────────────
        new_kv_by_pos: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        for p in range(len(prompt_ids)):
            layer_kvs = []
            for layer_idx in range(self.n_layers):
                k, v = kv_cache.snapshot_layer_kv(layer_idx, p + 1)
                # store as (H, 1, D) slice
                layer_kvs.append((k[:, p:p+1, :], v[:, p:p+1, :]))
            new_kv_by_pos.append(layer_kvs)

        radix_cache.insert(prompt_ids, new_kv_by_pos)
        radix_cache.release(prompt_ids[:prefix_len])

        return tokens
