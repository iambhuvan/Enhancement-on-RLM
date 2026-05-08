"""
kv_transformer.py — KV-Cache-aware transformer built on MiniTorch (h4).

Extends the standard MiniTorch DecoderLM with two extra inference modes:

  prefill(idx, cache)         — process a full prompt, fill KVCacheBuffer
  decode_step(tok, pos, cache)— decode one token using the filled cache

The standard forward(idx) path is unchanged and identical to DecoderLM,
so training and full-sequence evaluation work exactly as before.

Architecture
────────────
KVDecoderLM
  ├── token_embeddings    : Embedding(n_vocab,      n_embd)
  ├── position_embeddings : Embedding(n_positions,  n_embd)
  ├── dropout             : Dropout(p_dropout)
  ├── t_layer_1..4        : KVTransformerLayer  (Pre-LN, causal MHA)
  ├── ln                  : LayerNorm1d(n_embd)
  └── lm_head             : Linear(n_embd, n_vocab)

KVTransformerLayer
  ├── ln_1      : LayerNorm1d
  ├── attention : KVCacheAttention   ← modified to use KVCacheBuffer
  ├── ln_2      : LayerNorm1d
  └── ff        : FeedForward (unchanged)

KVCacheAttention
  Stores the logic for two compute paths:
    • full(x)           — standard O(T²) attention (training / no-cache eval)
    • decode(x, cache)  — O(T) attention using cached past K,V

MiniTorch tensor API notes
──────────────────────────
• No torch.cat — K,V concatenation is done via numpy and tensor_from_numpy.
• permute() may create non-contiguous tensor → always call .contiguous()
  before .view() or .to_numpy().
• Linear expects 2D input (batch, in_size) — reshape before and after.
• Embedding expects integer-valued float32 tensor; indices must be floats
  but are cast to int inside one_hot().
"""
from __future__ import annotations

import os
import sys
import numpy as np
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# MiniTorch path setup — points to the h4 assignment copy
# ---------------------------------------------------------------------------

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))

# Patch numba BEFORE minitorch import (NumPy ≥ 2.2 breaks numba)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from _numba_mock import install_mock as _install_mock  # noqa: E402
_install_mock()


def _find_h4_dir() -> str:
    """Walk up from this file until we find assignments/h4."""
    d = _THIS_DIR
    for _ in range(10):
        candidate = os.path.join(d, "assignments", "h4")
        if os.path.isdir(candidate):
            return candidate
        d = os.path.dirname(d)
    raise FileNotFoundError(
        "Could not locate assignments/h4 relative to "
        f"{_THIS_DIR}. Pass --h4_path explicitly."
    )


_H4_DIR = _find_h4_dir()
if _H4_DIR not in sys.path:
    sys.path.insert(0, _H4_DIR)

import minitorch
from minitorch import tensor_from_numpy, tensor
from minitorch.module import Module, Parameter
from minitorch.modules_basic import Linear, Embedding, LayerNorm1d, Dropout
from minitorch.modules_transfomer import FeedForward
from minitorch.nn import softmax
from minitorch.tensor_ops import TensorBackend

from kv_cache import KVCacheBuffer
from radix_trie import RadixAttentionCache

datatype = np.float32


# ---------------------------------------------------------------------------
# KVCacheAttention
# ---------------------------------------------------------------------------

class KVCacheAttention(Module):
    """
    Multi-Head Attention that supports both full (standard) and cached
    (incremental) forward passes.

    full path   : identical to MiniTorch MultiHeadAttention.
    decode path : only the new token's Q,K,V are projected; past K,V are
                  read from the KVCacheBuffer so attention is O(1) in
                  projections and O(T) in scores instead of O(T) + O(T²).
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        layer_idx: int,
        p_dropout: float = 0.0,
        bias: bool = True,
        backend: TensorBackend = None,
    ) -> None:
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

        self.backend         = backend
        self.n_embd          = n_embd
        self.n_head          = n_head
        self.layer_idx       = layer_idx
        self.attn_hidden_dim = n_embd // n_head

        self.q_projection   = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.k_projection   = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.v_projection   = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.out_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.dropout        = Dropout(p_dropout)

    # ------------------------------------------------------------------
    # Helpers shared by both paths
    # ------------------------------------------------------------------

    def _create_causal_mask(
        self, batch_size: int, n_head: int, seq_len: int
    ):
        mask = (
            -np.finfo(datatype).max
            * np.triu(
                np.ones((batch_size, n_head, seq_len, seq_len), dtype=datatype), 1
            )
        )
        return tensor_from_numpy(mask, backend=self.backend)

    def _project(self, x_2d, batch_size: int, seq_len: int):
        """Project x_2d → (Q, KT, V) in 4-D multi-head format."""
        head_dim = self.attn_hidden_dim
        n_head   = self.n_head

        q = (
            self.q_projection(x_2d)
            .view(batch_size, seq_len, n_head, head_dim)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k_projection(x_2d)
            .view(batch_size, seq_len, n_head, head_dim)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v_projection(x_2d)
            .view(batch_size, seq_len, n_head, head_dim)
            .permute(0, 2, 1, 3)
        )
        kT = k.permute(0, 1, 3, 2)  # (batch, n_head, head_dim, seq_len)
        return q, kT, v, k

    # ------------------------------------------------------------------
    # Full forward (standard, no cache)
    # ------------------------------------------------------------------

    def forward(self, x, kv_cache: Optional[KVCacheBuffer] = None):
        """
        If kv_cache is None  → standard causal self-attention (training).
        If kv_cache provided → cached incremental decode (see decode below).
        """
        if kv_cache is None:
            return self._forward_full(x)
        return self._forward_decode(x, kv_cache)

    def _forward_full(self, x):
        """Standard O(T²) causal self-attention — identical to MiniTorch MHA."""
        batch_size, seq_len, n_embd = x.shape
        x_2d = x.view(batch_size * seq_len, n_embd)

        q, kT, v, _ = self._project(x_2d, batch_size, seq_len)

        scores = (q @ kT) / np.sqrt(self.attn_hidden_dim)
        scores = scores + self._create_causal_mask(batch_size, self.n_head, seq_len)
        attn   = softmax(scores, dim=3)
        attn   = self.dropout(attn)
        out    = attn @ v  # (batch, n_head, seq_len, head_dim)

        out    = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, n_embd)
        result = self.out_projection(
            out.view(batch_size * seq_len, n_embd)
        ).view(batch_size, seq_len, n_embd)
        return result

    # ------------------------------------------------------------------
    # Cached decode forward (O(1) projections + O(T) attention)
    # ------------------------------------------------------------------

    def _forward_decode(self, x, kv_cache: KVCacheBuffer):
        """
        Incremental decode: x is (batch, 1, n_embd) — a single new token.

        Steps
        -----
        1. Project only the new token to Q, K_new, V_new.
        2. Convert K_new, V_new to numpy and append to KVCacheBuffer.
        3. Reconstruct full K, V tensor from the buffer (past + current).
        4. Compute attention Q_new × K_full^T (no causal mask needed:
           the new token can attend to all cached past positions).
        5. Project output and return.
        """
        batch_size, new_len, n_embd = x.shape
        assert new_len == 1, (
            "decode_step only processes one token at a time; got seq_len={new_len}"
        )
        head_dim = self.attn_hidden_dim
        n_head   = self.n_head

        x_2d = x.view(batch_size * new_len, n_embd)

        # 1. Project new token → Q (kept as tensor), K/V (converted to numpy)
        q = (
            self.q_projection(x_2d)
            .view(batch_size, new_len, n_head, head_dim)
            .permute(0, 2, 1, 3)
        )  # (batch, n_head, 1, head_dim)

        k_new_t = (
            self.k_projection(x_2d)
            .view(batch_size, new_len, n_head, head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # (batch, n_head, 1, head_dim)
        v_new_t = (
            self.v_projection(x_2d)
            .view(batch_size, new_len, n_head, head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        # 2. numpy arrays — breaks grad graph intentionally (inference only)
        k_new_np = k_new_t.to_numpy()   # (batch, n_head, 1, head_dim)
        v_new_np = v_new_t.to_numpy()

        # 3. Append to cache; get full K, V including new token
        k_full_np, v_full_np = kv_cache.update_and_get(
            self.layer_idx, k_new_np, v_new_np
        )

        # 4. Reconstruct tensors from numpy
        k_full  = tensor_from_numpy(k_full_np, backend=self.backend)
        v_full  = tensor_from_numpy(v_full_np, backend=self.backend)
        kT_full = k_full.permute(0, 1, 3, 2)  # (batch, n_head, head_dim, full_len)

        # 5. Attention: Q(1) × KT(full) — no causal mask (all past is valid)
        scores = (q @ kT_full) / np.sqrt(head_dim)
        attn   = softmax(scores, dim=3)
        attn   = self.dropout(attn)
        out    = attn @ v_full  # (batch, n_head, 1, head_dim)

        # 6. Reshape and output-project
        out    = out.permute(0, 2, 1, 3).contiguous().view(batch_size, new_len, n_embd)
        result = self.out_projection(
            out.view(batch_size * new_len, n_embd)
        ).view(batch_size, new_len, n_embd)
        return result


# ---------------------------------------------------------------------------
# KVTransformerLayer
# ---------------------------------------------------------------------------

class KVTransformerLayer(Module):
    """
    Pre-LN Transformer layer that delegates to KVCacheAttention.

    forward(x)             → standard path (training / full eval)
    forward(x, kv_cache)   → cached decode path
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        layer_idx: int,
        p_dropout: float = 0.0,
        ln_eps: float = 1e-8,
        bias: bool = True,
        backend: TensorBackend = None,
    ) -> None:
        super().__init__()
        self.n_embd    = n_embd
        self.layer_idx = layer_idx

        self.ln_1      = LayerNorm1d(dim=n_embd, eps=ln_eps, backend=backend)
        self.ln_2      = LayerNorm1d(dim=n_embd, eps=ln_eps, backend=backend)
        self.attention = KVCacheAttention(
            n_embd=n_embd,
            n_head=n_head,
            layer_idx=layer_idx,
            p_dropout=p_dropout,
            bias=bias,
            backend=backend,
        )
        self.ff = FeedForward(
            n_embd=n_embd,
            p_dropout=p_dropout,
            bias=bias,
            backend=backend,
        )

    def forward(self, x, kv_cache: Optional[KVCacheBuffer] = None):
        batch_size, seq_len, x_dim = x.shape

        # Pre-LN → Attention residual
        x_norm = (
            self.ln_1(x.view(batch_size * seq_len, x_dim))
            .view(batch_size, seq_len, x_dim)
        )
        x = x + self.attention(x_norm, kv_cache)

        # Pre-LN → FFN residual
        x_norm = (
            self.ln_2(x.view(batch_size * seq_len, x_dim))
            .view(batch_size, seq_len, x_dim)
        )
        x = x + self.ff(x_norm)
        return x


# ---------------------------------------------------------------------------
# KVDecoderLM
# ---------------------------------------------------------------------------

class KVDecoderLM(Module):
    """
    Decoder-only Pre-LN Transformer with KV cache inference support.

    Identical architecture to MiniTorch's DecoderLM (4 transformer layers,
    token + position embeddings, final LN + LM head) with three additional
    inference methods:

        prefill(idx, kv_cache)          — process prompt + fill cache
        decode_step(token_id, pos, cache)— one-token incremental decode
        generate(prompt_ids, max_new)    — full generation loop w/ cache

    The standard forward(idx) path is unchanged.

    Args:
        n_vocab     : Vocabulary size.
        n_embd      : Embedding / hidden dimension.
        n_head      : Number of attention heads.
        n_positions : Maximum sequence length.
        p_dropout   : Dropout probability (use 0.0 for inference).
        ln_eps      : LayerNorm epsilon.
        bias        : Add bias in linear layers.
        backend     : MiniTorch TensorBackend.
    """

    N_LAYERS = 4  # fixed to match DecoderLM

    def __init__(
        self,
        n_vocab: int,
        n_embd: int,
        n_head: int,
        n_positions: int,
        p_dropout: float = 0.0,
        ln_eps: float = 1e-5,
        bias: bool = True,
        backend: TensorBackend = None,
    ) -> None:
        super().__init__()

        self.backend     = backend
        self.n_embd      = n_embd
        self.n_vocab     = n_vocab
        self.n_positions = n_positions
        self.n_head      = n_head
        self.head_dim    = n_embd // n_head

        self.token_embeddings    = Embedding(n_vocab,      n_embd, backend=backend)
        self.position_embeddings = Embedding(n_positions,  n_embd, backend=backend)
        self.dropout             = Dropout(p_dropout)

        # Four transformer layers — named t_layer_N to match DecoderLM
        self.t_layer_1 = KVTransformerLayer(
            n_embd, n_head, layer_idx=0,
            p_dropout=p_dropout, ln_eps=ln_eps, bias=bias, backend=backend,
        )
        self.t_layer_2 = KVTransformerLayer(
            n_embd, n_head, layer_idx=1,
            p_dropout=p_dropout, ln_eps=ln_eps, bias=bias, backend=backend,
        )
        self.t_layer_3 = KVTransformerLayer(
            n_embd, n_head, layer_idx=2,
            p_dropout=p_dropout, ln_eps=ln_eps, bias=bias, backend=backend,
        )
        self.t_layer_4 = KVTransformerLayer(
            n_embd, n_head, layer_idx=3,
            p_dropout=p_dropout, ln_eps=ln_eps, bias=bias, backend=backend,
        )

        self.ln      = LayerNorm1d(dim=n_embd, eps=ln_eps, backend=backend)
        self.lm_head = Linear(n_embd, n_vocab, bias=bias, backend=backend)

    @property
    def _layers(self) -> List[KVTransformerLayer]:
        return [self.t_layer_1, self.t_layer_2, self.t_layer_3, self.t_layer_4]

    # ------------------------------------------------------------------
    # Standard forward (no cache)
    # ------------------------------------------------------------------

    def forward(self, idx):
        """
        Standard full-sequence forward pass (no KV cache).

        Args:
            idx : (batch_size, seq_len) integer float32 token tensor.

        Returns:
            logits : (batch_size, seq_len, n_vocab)
        """
        batch_size, seq_len = idx.shape
        pos = tensor(
            [float(i) for i in range(seq_len)], backend=self.backend
        ).view(1, seq_len)

        tok_emb = self.token_embeddings(idx)
        pos_emb = self.position_embeddings(pos).view(1, seq_len, self.n_embd)
        x       = self.dropout(tok_emb + pos_emb)

        for layer in self._layers:
            x = layer(x)  # no kv_cache

        x = self.ln(x.view(batch_size * seq_len, self.n_embd)).view(
            batch_size, seq_len, self.n_embd
        )
        logits = self.lm_head(
            x.view(batch_size * seq_len, self.n_embd)
        ).view(batch_size, seq_len, self.n_vocab)
        return logits

    # ------------------------------------------------------------------
    # Prefill — full prompt, fills KV cache
    # ------------------------------------------------------------------

    def prefill(self, idx, kv_cache: KVCacheBuffer):
        """
        Process the full prompt tensor and fill `kv_cache`.

        The KV projections for EVERY position are written to kv_cache so
        subsequent decode_step calls only project one new token at a time.

        Args:
            idx      : (1, seq_len) prompt token tensor.
            kv_cache : Empty KVCacheBuffer (will be filled in-place).

        Returns:
            logits : (1, seq_len, n_vocab) — logits for all prompt positions.
                     Use logits[:, -1, :] to sample the first generated token.
        """
        batch_size, seq_len = idx.shape
        pos = tensor(
            [float(i) for i in range(seq_len)], backend=self.backend
        ).view(1, seq_len)

        tok_emb = self.token_embeddings(idx)
        pos_emb = self.position_embeddings(pos).view(1, seq_len, self.n_embd)
        x       = tok_emb + pos_emb  # no dropout during inference

        # Run each layer — use the full-sequence path but capture K,V
        # by running a parallel decode loop that fills the cache.
        # We do this by reusing the cached-decode path layer-by-layer,
        # but processing ALL tokens at once for the initial fill.
        x = self._prefill_through_layers(x, kv_cache, seq_len)

        kv_cache.advance(seq_len)

        x = self.ln(x.view(batch_size * seq_len, self.n_embd)).view(
            batch_size, seq_len, self.n_embd
        )
        logits = self.lm_head(
            x.view(batch_size * seq_len, self.n_embd)
        ).view(batch_size, seq_len, self.n_vocab)
        return logits

    def _prefill_through_layers(self, x, kv_cache: KVCacheBuffer, seq_len: int):
        """
        Run the full-sequence input through all layers while populating the
        KV cache with the computed K,V for every position.

        The cache is written using the same update_and_get path as decode so
        that cached K,V are layout-compatible with subsequent decode steps.
        """
        batch_size = x.shape[0]
        n_embd     = self.n_embd

        for layer in self._layers:
            layer_idx = layer.layer_idx
            attn      = layer.attention

            # --- LayerNorm before attention ---
            x_norm = (
                layer.ln_1(x.view(batch_size * seq_len, n_embd))
                .view(batch_size, seq_len, n_embd)
            )

            # --- Project ALL tokens to Q, K, V ---
            x_2d   = x_norm.view(batch_size * seq_len, n_embd)
            k_all  = (
                attn.k_projection(x_2d)
                .view(batch_size, seq_len, attn.n_head, attn.attn_hidden_dim)
                .permute(0, 2, 1, 3)
                .contiguous()
            )  # (batch, n_head, seq_len, head_dim)
            v_all  = (
                attn.v_projection(x_2d)
                .view(batch_size, seq_len, attn.n_head, attn.attn_hidden_dim)
                .permute(0, 2, 1, 3)
                .contiguous()
            )

            # --- Store ALL K,V in cache (breaks grad graph — inference only) ---
            k_np = k_all.to_numpy()   # (batch, n_head, seq_len, head_dim)
            v_np = v_all.to_numpy()
            # update_and_get expects n_new dimension at axis 2
            kv_cache.update_and_get(layer_idx, k_np, v_np)

            # --- Standard causal attention for the full prompt ---
            q_all  = (
                attn.q_projection(x_2d)
                .view(batch_size, seq_len, attn.n_head, attn.attn_hidden_dim)
                .permute(0, 2, 1, 3)
            )
            kT_all = k_all.permute(0, 1, 3, 2)
            scores = (q_all @ kT_all) / np.sqrt(attn.attn_hidden_dim)
            mask   = attn._create_causal_mask(batch_size, attn.n_head, seq_len)
            scores = scores + mask
            att_w  = softmax(scores, dim=3)
            out    = att_w @ v_all  # (batch, n_head, seq_len, head_dim)

            out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, n_embd)
            attn_out = attn.out_projection(
                out.view(batch_size * seq_len, n_embd)
            ).view(batch_size, seq_len, n_embd)

            x = x + attn_out

            # --- FFN ---
            x_norm2 = (
                layer.ln_2(x.view(batch_size * seq_len, n_embd))
                .view(batch_size, seq_len, n_embd)
            )
            x = x + layer.ff(x_norm2)

        return x

    # ------------------------------------------------------------------
    # Single-token decode step
    # ------------------------------------------------------------------

    def decode_step(
        self,
        token_id: float,
        pos: int,
        kv_cache: KVCacheBuffer,
    ):
        """
        Decode one new token using the filled KV cache.

        Args:
            token_id : Float value of the new token's ID (MiniTorch uses float32).
            pos      : Absolute position of this token in the sequence.
            kv_cache : KVCacheBuffer with past K,V already filled by prefill().

        Returns:
            logits : (1, 1, n_vocab) — logits for the next token.
        """
        idx = tensor([token_id], backend=self.backend).view(1, 1)
        pos_t = tensor([float(pos)], backend=self.backend).view(1, 1)

        tok_emb = self.token_embeddings(idx)                          # (1, 1, n_embd)
        pos_emb = self.position_embeddings(pos_t).view(1, 1, self.n_embd)
        x       = tok_emb + pos_emb                                   # (1, 1, n_embd)

        # Each layer uses the cached decode path
        for layer in self._layers:
            x = layer(x, kv_cache=kv_cache)

        # Advance cache after ALL layers have processed this token
        kv_cache.advance(1)

        x = self.ln(x.view(1, self.n_embd)).view(1, 1, self.n_embd)
        logits = self.lm_head(x.view(1, self.n_embd)).view(1, 1, self.n_vocab)
        return logits

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    def make_kv_cache(self, max_seq_len: int) -> KVCacheBuffer:
        """Create a KVCacheBuffer sized for this model."""
        return KVCacheBuffer(
            n_layers=self.N_LAYERS,
            n_heads=self.n_head,
            head_dim=self.head_dim,
            max_seq_len=max_seq_len,
        )

    def generate_no_cache(
        self,
        prompt_ids: List[int],
        max_new_tokens: int,
    ) -> List[int]:
        """
        Baseline greedy generation WITHOUT KV cache.

        At each step t the model re-processes ALL tokens [0..t-1] to get
        the logit for position t.  This is O(T²) in attention and O(T)
        in K,V projections per step → O(n³) total for n new tokens.

        Returns the full token sequence (prompt + generated tokens).
        """
        self.eval()
        tokens = list(prompt_ids)

        for _ in range(max_new_tokens):
            idx_t = tensor(
                [float(t) for t in tokens], backend=self.backend
            ).view(1, len(tokens))
            logits = self.forward(idx_t)       # (1, T, n_vocab)
            # Greedy: argmax at last position
            last_logits = logits.to_numpy()[0, -1, :]  # (n_vocab,)
            next_tok    = int(np.argmax(last_logits))
            tokens.append(next_tok)

        return tokens

    def generate_with_cache(
        self,
        prompt_ids: List[int],
        max_new_tokens: int,
        kv_cache: Optional[KVCacheBuffer] = None,
    ) -> List[int]:
        """
        Greedy generation WITH KV cache.

        Prefill processes the prompt once (O(T_prompt²) attention, O(T_prompt)
        projections).  Each new token only requires O(1) projections + O(T)
        attention → total O(n²) vs O(n³) without cache.

        Returns the full token sequence (prompt + generated tokens).
        """
        self.eval()
        if kv_cache is None:
            kv_cache = self.make_kv_cache(
                max_seq_len=len(prompt_ids) + max_new_tokens + 1
            )

        # Prefill
        prompt_t = tensor(
            [float(t) for t in prompt_ids], backend=self.backend
        ).view(1, len(prompt_ids))
        prefill_logits = self.prefill(prompt_t, kv_cache)  # (1, T_p, n_vocab)

        # First new token from prefill logits
        last_logits = prefill_logits.to_numpy()[0, -1, :]
        next_tok    = int(np.argmax(last_logits))
        tokens      = list(prompt_ids) + [next_tok]

        # Decode remaining tokens
        for step in range(1, max_new_tokens):
            pos     = len(prompt_ids) + step - 1
            logits  = self.decode_step(float(next_tok), pos, kv_cache)
            last_logits = logits.to_numpy()[0, 0, :]
            next_tok    = int(np.argmax(last_logits))
            tokens.append(next_tok)

        return tokens

    def generate_with_radix(
        self,
        prompt_ids: List[int],
        max_new_tokens: int,
        radix_cache: RadixAttentionCache,
    ) -> List[int]:
        """
        Greedy generation using RadixAttentionCache for prefix sharing.

        CORRECTNESS PROPERTY: decode_step for a position i (with cache
        containing positions 0..i-1) computes K,V identical to a full
        prefill, because at every layer the K,V projection input is the
        same hidden state (proven by induction over layers & positions).
        Therefore this path produces the same tokens as generate_with_cache.


        Before running prefill, looks up the longest matching prefix in the
        radix trie and restores those K,V blocks into the KVCacheBuffer,
        skipping their recomputation.  After generation, inserts all K,V
        blocks back into the trie for future requests.

        Returns the full token sequence (prompt + generated tokens).
        """
        self.eval()
        total_len = len(prompt_ids) + max_new_tokens + 1
        kv_cache  = self.make_kv_cache(total_len)

        # 1. Look up prefix in radix trie
        prefix_len, kv_list = radix_cache.lookup_prefix(prompt_ids)

        # 2. Restore K,V for cached prefix — concatenate per-position slices
        #    into full blocks, one per layer.  Set pointer to prefix_len-1 so
        #    we re-process the boundary token to get its correct cross-attended
        #    K,V before continuing into the suffix.
        if prefix_len > 0:
            for layer_idx in range(self.N_LAYERS):
                # kv_list[pos][layer_idx] = (k_np, v_np), shape (1,heads,1,head_dim)
                k_prefix = np.concatenate(
                    [kv_list[pos][layer_idx][0] for pos in range(prefix_len)], axis=2
                )  # (1, heads, prefix_len, head_dim)
                v_prefix = np.concatenate(
                    [kv_list[pos][layer_idx][1] for pos in range(prefix_len)], axis=2
                )
                kv_cache.restore_layer_kv(layer_idx, k_prefix, v_prefix, prefix_len)
            # Back up one position: re-run boundary token so all layers see
            # a consistent hidden state at the prefix/suffix junction.
            kv_cache.current_len = max(0, prefix_len - 1)

        # 3. Process each prompt token from decode_start onward via decode_step.
        #    decode_step cross-attends to the already-restored prefix K,V so
        #    suffix tokens see full context without recomputing the prefix.
        #    The last iteration returns logits for the first generated token.
        decode_start = max(0, prefix_len - 1) if prefix_len > 0 else 0
        last_logit: Optional[object] = None
        for abs_pos in range(decode_start, len(prompt_ids)):
            last_logit = self.decode_step(
                float(prompt_ids[abs_pos]), abs_pos, kv_cache
            )

        # 4. First generated token
        next_tok = int(np.argmax(last_logit.to_numpy()[0, 0, :]))  # type: ignore[union-attr]
        tokens   = list(prompt_ids) + [next_tok]

        # 5. Continue generating
        for step in range(1, max_new_tokens):
            pos      = len(prompt_ids) + step - 1
            logits   = self.decode_step(float(next_tok), pos, kv_cache)
            next_tok = int(np.argmax(logits.to_numpy()[0, 0, :]))
            tokens.append(next_tok)

        # 6. Insert full prompt K,V into radix trie for future prefix sharing
        new_per_pos_kv = self._snapshot_prefix_kv(kv_cache, prompt_ids)
        radix_cache.insert(prompt_ids, new_per_pos_kv)
        radix_cache.release(prompt_ids[:prefix_len])

        return tokens

    # ------------------------------------------------------------------
    # Internal helpers for RadixAttention
    # ------------------------------------------------------------------

    def _snapshot_prefix_kv(
        self, kv_cache: KVCacheBuffer, token_ids: List[int]
    ) -> List[List]:
        """
        Build the per-position KV list from a filled KVCacheBuffer for
        insertion into RadixAttentionCache.
        """
        per_pos: List[List] = []
        for pos in range(len(token_ids)):
            layer_kvs = []
            for layer_idx in range(self.N_LAYERS):
                k_full, v_full = kv_cache.get_layer_kv(layer_idx)
                # Extract single position slice
                k_pos = k_full[:, :, pos : pos + 1, :].copy()
                v_pos = v_full[:, :, pos : pos + 1, :].copy()
                layer_kvs.append((k_pos, v_pos))
            per_pos.append(layer_kvs)
        return per_pos
