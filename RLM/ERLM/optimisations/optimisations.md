# RLM Upgrade Ideas: Deep-Dive Implementation Plan

## Table of Contents

1. [Idea 1: Hierarchical KV Cache with Compression](#idea-1-hierarchical-kv-cache-with-compression)
2. [Idea 2: Learned Recursion Policy via Reinforcement Learning](#idea-2-learned-recursion-policy-via-reinforcement-learning)
3. [Idea 3: RLM for Code — Recursive Code Oracle](#idea-3-rlm-for-code--recursive-code-oracle)
4. [Idea 4: Distilling RLM Traces into Direct Long-Context Models](#idea-4-distilling-rlm-traces-into-direct-long-context-models)
5. [Idea 5: Multi-Resolution Recursive Attention](#idea-5-multi-resolution-recursive-attention)
6. [Idea 6: Cross-Query Caching for RLM Serving](#idea-6-cross-query-caching-for-rlm-serving)
7. [Recommended Combination Strategy](#recommended-combination-strategy)

---

## Idea 1: Hierarchical KV Cache with Compression

### Core Insight

In standard LLM inference, every request starts fresh. In RLM, recursive subcalls form a **tree** where parent context is shared across children, siblings share the same decomposition prompt, and grandparent context is shared even more broadly. Yet current systems recompute KV caches from scratch for every subcall. This is massively wasteful.

**Key observation:** As you go deeper in the recursion tree, ancestor context becomes progressively less relevant to the specific subcall — it can be compressed more aggressively without quality loss.

### Literature Foundation

| Paper | Key Technique | Relevance |
|---|---|---|
| **DeFT** (ICLR 2025 Spotlight, arxiv 2404.00242) | Flash Tree-Attention with KV-Guided Grouping | Purpose-built for tree-structured inference; reduces 73-99% KV cache IO; groups queries by shared prefix path |
| **RadixAttention / SGLang** (arxiv 2312.07104) | Radix tree for KV cache management | Stores all KV caches in a compressed trie; up to 6.4x throughput; cache-aware scheduling maximizes hit rate |
| **PyramidKV** (TMLR 2025) | Layer-wise budget allocation via arithmetic progression | Lower layers get more budget (broad context), upper layers get less (focused); matches full KV at 12% retention |
| **SqueezeAttention** (ICLR 2025) | Joint sequence-layer KV compression | 30-70% memory reduction; 2.2x throughput; orthogonal to token-level methods |
| **KVQuant** (NeurIPS 2024) | Per-channel pre-RoPE key quantization + non-uniform codebooks | 8x compression at 2-bit; enables 1M context on single A100 |
| **H2O: Heavy-Hitter Oracle** (NeurIPS 2023) | Attention-score-based token eviction | Only 5% of KV cache needed; up to 20x reduction |
| **MLA / DeepSeek-V2** | Low-rank latent compression of K,V | 93.3% KV cache reduction; 5.76x throughput vs MHA |
| **LMCache** (arxiv 2510.09665) | Multi-tier KV cache storage (GPU/CPU/disk/remote) | Up to 15x throughput; chunk-level (not just prefix) reuse |
| **CacheBlend** (EuroSys 2025 Best Paper) | Non-prefix KV cache reuse with selective recomputation | 2.2-3.3x TTFT reduction; reuses KV at arbitrary positions |
| **Mooncake** (FAST 2025 Best Paper, arxiv 2407.00079) | Disaggregated prefill/decode with distributed KV pool | 525% throughput increase in long-context; RDMA-based KV transfer |

### Architecture Design

```
┌─────────────────────────────────────────────────────────────┐
│                HIERARCHICAL KV CACHE SYSTEM                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Recursion Tree:          KV Cache Tree (mirrors it):        │
│                                                              │
│       Root                    [Full precision KV]            │
│      /    \                   /                \             │
│    S1      S2          [Compressed KV]    [Compressed KV]    │
│   / \       \          /       \                \            │
│  S1a S1b    S2a   [Quantized]  [Quantized]  [Quantized]     │
│                                                              │
│  COMPRESSION PER DEPTH:                                      │
│  ─────────────────────────────────────────────────           │
│  Depth 0: FP16, all heads, all tokens (full fidelity)        │
│  Depth 1: FP8, top-75% heads, top-80% tokens (H2O pruning)  │
│  Depth 2: INT4, top-50% heads, top-50% tokens                │
│  Depth 3+: INT2, top-25% heads, top-25% tokens               │
│                                                              │
│  LAYER-WISE BUDGET (PyramidKV-inspired):                     │
│  ─────────────────────────────────────────────────           │
│  Lower layers (0-10):  80% budget  (broad context capture)   │
│  Middle layers (11-20): 50% budget  (intermediate features)  │
│  Upper layers (21-31):  20% budget  (task-specific, compress)│
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Detailed Implementation

```python
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class CompressionLevel(Enum):
    FULL = 0        # FP16, all heads, all tokens
    LIGHT = 1       # FP8, top-75% heads, H2O top-80% tokens
    MEDIUM = 2      # INT4, top-50% heads, top-50% tokens
    AGGRESSIVE = 3  # INT2, top-25% heads, top-25% tokens


@dataclass
class KVCacheNode:
    """A single node in the KV cache tree, corresponding to one RLM subcall."""
    node_id: str
    parent_id: Optional[str]
    depth: int
    keys: torch.Tensor          # [n_layers, n_heads, seq_len, head_dim]
    values: torch.Tensor        # [n_layers, n_heads, seq_len, head_dim]
    compression_level: CompressionLevel
    attention_scores: torch.Tensor  # [n_layers, n_heads, seq_len] — cumulative
    head_importance: torch.Tensor   # [n_layers, n_heads] — for head pruning
    token_mask: torch.Tensor        # [n_layers, seq_len] — which tokens are retained
    children: List[str] = field(default_factory=list)

    @property
    def memory_bytes(self) -> int:
        bits_per_element = {
            CompressionLevel.FULL: 16,
            CompressionLevel.LIGHT: 8,
            CompressionLevel.MEDIUM: 4,
            CompressionLevel.AGGRESSIVE: 2,
        }[self.compression_level]
        active_tokens = self.token_mask.sum().item()
        active_heads = (self.head_importance > 0).sum().item()
        return int(active_tokens * active_heads * self.keys.shape[-1] * bits_per_element / 8 * 2)


class PyramidLayerBudget:
    """
    Allocate KV cache budget per layer using arithmetic progression.
    Inspired by PyramidKV (TMLR 2025): lower layers capture broad context
    and need more budget; upper layers are task-specific and can be compressed.
    """

    def __init__(self, n_layers: int, total_budget_ratio: float = 0.5):
        self.n_layers = n_layers
        self.total_budget_ratio = total_budget_ratio

    def get_layer_budgets(self, seq_len: int) -> List[int]:
        """Return per-layer token budget as arithmetic progression."""
        # Highest budget for layer 0, lowest for layer n-1
        max_budget = int(seq_len * min(1.0, self.total_budget_ratio * 2))
        min_budget = max(1, int(seq_len * self.total_budget_ratio * 0.1))

        budgets = []
        for i in range(self.n_layers):
            # Linear interpolation from max to min
            ratio = i / max(1, self.n_layers - 1)
            budget = int(max_budget * (1 - ratio) + min_budget * ratio)
            budgets.append(budget)
        return budgets


class HeavyHitterTokenSelector:
    """
    H2O-inspired token selection: keep heavy-hitter tokens (high cumulative
    attention) plus recent tokens. Operates per-layer, per-head.
    """

    def __init__(self, recent_window: int = 64):
        self.recent_window = recent_window

    def select_tokens(
        self,
        attention_scores: torch.Tensor,  # [n_heads, seq_len]
        budget: int,
    ) -> torch.Tensor:
        """Return boolean mask of tokens to keep."""
        seq_len = attention_scores.shape[-1]
        if budget >= seq_len:
            return torch.ones(seq_len, dtype=torch.bool, device=attention_scores.device)

        mask = torch.zeros(seq_len, dtype=torch.bool, device=attention_scores.device)

        # Always keep recent tokens
        recent_start = max(0, seq_len - self.recent_window)
        mask[recent_start:] = True
        remaining_budget = budget - mask.sum().item()

        if remaining_budget > 0:
            # Average attention across heads for token importance
            avg_scores = attention_scores.mean(dim=0)
            # Zero out already-selected tokens
            avg_scores[mask] = -float('inf')
            # Select top-k by attention score
            _, top_indices = avg_scores.topk(int(remaining_budget))
            mask[top_indices] = True

        return mask


class HeadPruner:
    """
    Prune attention heads based on importance scoring.
    Uses attention entropy: heads with very uniform attention (high entropy)
    carry less information and can be pruned first.
    """

    def compute_head_importance(
        self,
        attention_scores: torch.Tensor,  # [n_layers, n_heads, seq_len]
    ) -> torch.Tensor:
        """Return per-head importance score [n_layers, n_heads]."""
        # Normalize to probability distribution
        probs = F.softmax(attention_scores, dim=-1)
        # Compute entropy per head
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
        # Low entropy = focused attention = more important
        # Invert: importance = max_entropy - entropy
        max_entropy = torch.log(torch.tensor(attention_scores.shape[-1], dtype=torch.float32))
        importance = max_entropy - entropy
        return importance

    def get_head_mask(
        self,
        importance: torch.Tensor,  # [n_layers, n_heads]
        keep_ratio: float,
    ) -> torch.Tensor:
        """Return boolean mask [n_layers, n_heads] of heads to keep."""
        n_keep = max(1, int(importance.shape[1] * keep_ratio))
        mask = torch.zeros_like(importance, dtype=torch.bool)
        for layer_idx in range(importance.shape[0]):
            _, top_heads = importance[layer_idx].topk(n_keep)
            mask[layer_idx, top_heads] = True
        return mask


class KVQuantizer:
    """
    KVQuant-inspired quantization with per-channel pre-RoPE key quantization
    and non-uniform codebook for extreme compression.
    """

    @staticmethod
    def quantize(
        tensor: torch.Tensor,
        bits: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize tensor to specified bit-width. Returns (quantized, scale, zero_point)."""
        # Per-channel quantization (last dim = head_dim)
        min_val = tensor.amin(dim=-1, keepdim=True)
        max_val = tensor.amax(dim=-1, keepdim=True)

        n_levels = 2 ** bits
        scale = (max_val - min_val) / (n_levels - 1)
        scale = scale.clamp(min=1e-10)
        zero_point = (-min_val / scale).round()

        quantized = ((tensor - min_val) / scale).round().clamp(0, n_levels - 1)

        if bits <= 4:
            quantized = quantized.to(torch.uint8)
        elif bits <= 8:
            quantized = quantized.to(torch.int8)

        return quantized, scale, zero_point

    @staticmethod
    def dequantize(
        quantized: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
    ) -> torch.Tensor:
        """Dequantize back to float."""
        return (quantized.float() - zero_point) * scale


class HierarchicalKVCacheManager:
    """
    Main manager for tree-structured KV cache with depth-aware compression.

    Combines:
    - DeFT-style tree-aware prefix sharing
    - PyramidKV layer-wise budget allocation
    - H2O heavy-hitter token selection
    - Head pruning via attention entropy
    - KVQuant quantization at configurable bit-widths
    """

    # Compression configs per depth level
    DEPTH_CONFIGS = {
        0: {"head_keep_ratio": 1.0, "token_budget_ratio": 1.0, "quant_bits": 16},
        1: {"head_keep_ratio": 0.75, "token_budget_ratio": 0.8, "quant_bits": 8},
        2: {"head_keep_ratio": 0.50, "token_budget_ratio": 0.5, "quant_bits": 4},
        3: {"head_keep_ratio": 0.25, "token_budget_ratio": 0.25, "quant_bits": 2},
    }

    def __init__(
        self,
        n_layers: int = 32,
        n_heads: int = 32,
        head_dim: int = 128,
        max_gpu_memory_gb: float = 8.0,
    ):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_gpu_memory = int(max_gpu_memory_gb * 1024**3)

        self.cache_tree: Dict[str, KVCacheNode] = {}
        self.pyramid_budget = PyramidLayerBudget(n_layers)
        self.token_selector = HeavyHitterTokenSelector()
        self.head_pruner = HeadPruner()
        self.quantizer = KVQuantizer()

    def store_kv(
        self,
        node_id: str,
        parent_id: Optional[str],
        depth: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_scores: torch.Tensor,
    ) -> KVCacheNode:
        """Store KV cache for a subcall, applying depth-appropriate compression."""
        config = self.DEPTH_CONFIGS.get(depth, self.DEPTH_CONFIGS[3])

        # 1. Compute head importance and prune
        head_importance = self.head_pruner.compute_head_importance(attention_scores)
        head_mask = self.head_pruner.get_head_mask(head_importance, config["head_keep_ratio"])

        # 2. Compute per-layer token budgets (PyramidKV)
        seq_len = keys.shape[2]
        layer_budgets = self.pyramid_budget.get_layer_budgets(seq_len)
        layer_budgets = [int(b * config["token_budget_ratio"]) for b in layer_budgets]

        # 3. Select tokens per layer using H2O
        token_mask = torch.zeros(self.n_layers, seq_len, dtype=torch.bool, device=keys.device)
        for layer_idx in range(self.n_layers):
            layer_attn = attention_scores[layer_idx]  # [n_heads, seq_len]
            token_mask[layer_idx] = self.token_selector.select_tokens(
                layer_attn, layer_budgets[layer_idx]
            )

        # 4. Apply head and token masks
        compressed_keys = keys.clone()
        compressed_values = values.clone()
        for layer_idx in range(self.n_layers):
            # Zero out pruned heads
            pruned_heads = ~head_mask[layer_idx]
            compressed_keys[layer_idx, pruned_heads] = 0
            compressed_values[layer_idx, pruned_heads] = 0
            # Zero out pruned tokens
            pruned_tokens = ~token_mask[layer_idx]
            compressed_keys[layer_idx, :, pruned_tokens] = 0
            compressed_values[layer_idx, :, pruned_tokens] = 0

        # 5. Quantize if needed
        if config["quant_bits"] < 16:
            compressed_keys, k_scale, k_zp = self.quantizer.quantize(
                compressed_keys, config["quant_bits"]
            )
            compressed_values, v_scale, v_zp = self.quantizer.quantize(
                compressed_values, config["quant_bits"]
            )

        # 6. Create and store cache node
        node = KVCacheNode(
            node_id=node_id,
            parent_id=parent_id,
            depth=depth,
            keys=compressed_keys,
            values=compressed_values,
            compression_level=CompressionLevel(min(depth, 3)),
            attention_scores=attention_scores,
            head_importance=head_importance,
            token_mask=token_mask,
        )

        if parent_id and parent_id in self.cache_tree:
            self.cache_tree[parent_id].children.append(node_id)

        self.cache_tree[node_id] = node
        self._evict_if_needed()
        return node

    def get_cache_for_subcall(
        self,
        node_id: str,
        parent_id: str,
    ) -> Optional[torch.Tensor]:
        """
        Build KV cache for a new subcall by combining compressed ancestor caches.
        The parent's KV cache is reused (not recomputed), with progressive
        compression for older ancestors.
        """
        # Gather ancestor chain: [root, ..., grandparent, parent]
        ancestors = []
        current = parent_id
        while current is not None and current in self.cache_tree:
            ancestors.append(self.cache_tree[current])
            current = self.cache_tree[current].parent_id
        ancestors.reverse()

        if not ancestors:
            return None

        # Combine: oldest ancestors most compressed, parent least compressed
        combined_keys = []
        combined_values = []
        for ancestor in ancestors:
            combined_keys.append(ancestor.keys)
            combined_values.append(ancestor.values)

        # Concatenate along sequence dimension
        return torch.cat(combined_keys, dim=2), torch.cat(combined_values, dim=2)

    def _evict_if_needed(self):
        """LRU eviction when GPU memory budget is exceeded."""
        total_memory = sum(node.memory_bytes for node in self.cache_tree.values())
        if total_memory <= self.max_gpu_memory:
            return

        # Evict deepest, oldest nodes first (leaf nodes with no active children)
        leaf_nodes = [
            nid for nid, node in self.cache_tree.items()
            if not node.children
        ]
        # Sort by depth (deepest first), then by insertion order
        leaf_nodes.sort(key=lambda nid: -self.cache_tree[nid].depth)

        for nid in leaf_nodes:
            if total_memory <= self.max_gpu_memory:
                break
            total_memory -= self.cache_tree[nid].memory_bytes
            # Remove from parent's children list
            parent_id = self.cache_tree[nid].parent_id
            if parent_id and parent_id in self.cache_tree:
                self.cache_tree[parent_id].children.remove(nid)
            del self.cache_tree[nid]

    def get_memory_stats(self) -> dict:
        """Return memory usage statistics."""
        total = sum(n.memory_bytes for n in self.cache_tree.values())
        by_depth = {}
        for node in self.cache_tree.values():
            by_depth.setdefault(node.depth, 0)
            by_depth[node.depth] += node.memory_bytes
        return {
            "total_bytes": total,
            "total_mb": total / (1024**2),
            "num_nodes": len(self.cache_tree),
            "by_depth_mb": {d: b / (1024**2) for d, b in by_depth.items()},
        }
```

### Expected Compression Ratios

| Depth | Head Pruning | Token Pruning (H2O) | Layer Budget (Pyramid) | Quantization | Combined Compression |
|---|---|---|---|---|---|
| 0 | 1.0x | 1.0x | 1.0x | 1.0x (FP16) | **1.0x** (baseline) |
| 1 | 1.33x | 1.25x | 0.8x factor | 2.0x (FP8) | **~2.7x** |
| 2 | 2.0x | 2.0x | 0.5x factor | 4.0x (INT4) | **~8x** |
| 3 | 4.0x | 4.0x | 0.25x factor | 8.0x (INT2) | **~32x** |

**For a depth-3 recursion tree with 4 subcalls per level (64 leaf nodes):**
- Naive: 64 * full_cache = 64x memory
- Hierarchical: root(1x) + 4(0.37x) + 16(0.125x) + 64(0.03x) = ~5.4x memory
- **Savings: ~12x reduction** vs naive recomputation

### Key Evaluation Metrics

1. **KV memory per subcall** vs depth (should decrease exponentially)
2. **End-to-end latency** vs naive (no sharing) and prefix-only sharing
3. **Quality degradation** vs full-precision baseline (measure on RULER, needle-in-haystack)
4. **Throughput** (subcalls per second) at various recursion depths
5. Pareto curve: quality vs memory budget

### Connection to Course HW

- **HW4**: CUDA-accelerated attention kernels — extend to handle compressed/quantized KV inputs
- **HW3**: Transformer architecture — modify attention to accept hierarchical cache inputs
- **HW5**: Distributed systems — KV cache partitioning across GPUs for deep recursion trees

---

## Idea 2: Learned Recursion Policy via Reinforcement Learning

### Core Insight

RLM uses fixed heuristics for recursion decisions: hardcoded max depth, uniform fan-out, fixed chunking. But the optimal recursion strategy depends heavily on the query type, document structure, and remaining budget. A **learned policy** can dynamically decide when to recurse, how many subcalls to make, and when to stop — learning from experience which strategies work.

### Literature Foundation

| Paper | Key Technique | Relevance |
|---|---|---|
| **DeepSeek-R1** (arxiv 2501.12948) | GRPO (Group Relative Policy Optimization) | Critic-free RL; sample 16 completions per prompt, compute group-relative advantages; emergent self-correction |
| **Scaling Test-Time Compute** (Snell et al., 2024, arxiv 2408.03314) | Compute-optimal inference strategies | No single strategy dominates; optimal TTS scales with compute budget; smaller model + more compute can beat larger model |
| **IBPO** (arxiv 2501.17974) | Inference Budget-Constrained Policy Optimization | Formulates adaptive reasoning as utility maximization under budget constraint; models learn difficulty-aware allocation |
| **Mixture-of-Recursions (MoR)** (NeurIPS 2025, arxiv 2507.10524) | Per-token recursion depth routing | Lightweight MLP routers decide continue/halt per token per step; up to 2x throughput; end-to-end differentiable |
| **ReMoE** (ICLR 2025) | Fully differentiable MoE routing via sigmoid relaxation | Continuous relaxation of discrete routing; progressive sparsification schedule |
| **LayerSkip** (Meta, 2024) | Depth-dependent layer dropout + early exit | Enables accurate early exit at earlier layers; self-speculative decoding |
| **BudgetThinker** (arxiv 2508.17196) | Budget control tokens injected during generation | Model learns to modulate reasoning depth based on remaining budget signals; +4.9% accuracy with precise budget adherence |
| **TreeRL** (ACL 2025) | On-policy tree search with entropy-guided branching | Branch at high-uncertainty tokens; dense on-policy process rewards without separate PRM |
| **ReST-MCTS*** (NeurIPS 2024) | Self-training via process reward-guided tree search | MCTS generates traces; step-level rewards estimated by rollout probability; no human annotations needed |
| **BAM / E3 Metric** (EMNLP 2024) | Budget Allocation Model + correctness-computation trade-off metric | Provides principled evaluation framework for budget-aware reasoning |

### Architecture Design

```
┌──────────────────────────────────────────────────────────────┐
│                  LEARNED RECURSION POLICY                     │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  STATE SPACE:                                                 │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ • current_depth (int)                                 │    │
│  │ • remaining_budget (tokens, calls, depth-left)        │    │
│  │ • text_span_stats:                                    │    │
│  │   - length (tokens)                                   │    │
│  │   - entropy (vocabulary diversity)                    │    │
│  │   - keyword_density (query terms per token)           │    │
│  │   - structural_signals (headers, lists, code blocks)  │    │
│  │ • question_embedding (from LLM hidden state)          │    │
│  │ • parent_result_summary (if available)                │    │
│  │ • sibling_results (if any completed)                  │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                               │
│  ACTION SPACE:                                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ • ANSWER_NOW — stop recursing, answer from context    │    │
│  │ • RECURSE(n_splits, strategy):                        │    │
│  │   - n_splits ∈ {2, 4, 8, 16}                         │    │
│  │   - strategy ∈ {uniform, paragraph, semantic, header} │    │
│  │ • ZOOM(region_idx) — focus on specific region         │    │
│  │ • SEARCH(query) — use index to find relevant chunks   │    │
│  │ • BACKTRACK — abandon this branch, "no answer found"  │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                               │
│  REWARD FUNCTION:                                             │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ R = quality_score                                     │    │
│  │   - λ_tokens * (tokens_used / budget)                 │    │
│  │   - λ_calls * (calls_used / max_calls)                │    │
│  │   + bonus * early_correct_answer                      │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Detailed Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


# ─────────────────────────────────────────────
# State Representation
# ─────────────────────────────────────────────

@dataclass
class RecursionState:
    """Full state at a recursion decision point."""
    current_depth: int
    max_depth: int
    remaining_budget_tokens: int
    remaining_budget_calls: int
    text_span_length: int          # tokens in current span
    text_entropy: float            # vocabulary diversity
    keyword_density: float         # query-term frequency in span
    n_headers: int                 # structural signals
    n_code_blocks: int
    question_embedding: torch.Tensor  # [hidden_dim] from LLM
    parent_confidence: float       # parent's answer confidence (0-1)
    n_siblings_completed: int
    sibling_avg_confidence: float

    def to_tensor(self) -> torch.Tensor:
        """Flatten state to fixed-size vector."""
        scalar_features = torch.tensor([
            self.current_depth / self.max_depth,
            self.remaining_budget_tokens / 50000,
            self.remaining_budget_calls / 20,
            self.text_span_length / 10000,
            self.text_entropy,
            self.keyword_density,
            self.n_headers / 20,
            self.n_code_blocks / 10,
            self.parent_confidence,
            self.n_siblings_completed / 10,
            self.sibling_avg_confidence,
        ], dtype=torch.float32)
        return torch.cat([scalar_features, self.question_embedding.flatten()])


# ─────────────────────────────────────────────
# Action Space
# ─────────────────────────────────────────────

class ActionType:
    ANSWER_NOW = 0
    RECURSE_2 = 1
    RECURSE_4 = 2
    RECURSE_8 = 3
    ZOOM = 4
    SEARCH = 5
    BACKTRACK = 6
    NUM_ACTIONS = 7


SPLIT_STRATEGIES = ["uniform", "paragraph", "semantic", "header"]


@dataclass
class Action:
    action_type: int
    split_strategy: Optional[str] = None
    zoom_region: Optional[int] = None
    search_query: Optional[str] = None


# ─────────────────────────────────────────────
# Policy Network
# ─────────────────────────────────────────────

class RecursionPolicyNetwork(nn.Module):
    """
    Actor-Critic network for recursion decisions.

    Architecture inspired by MoR (Mixture-of-Recursions):
    lightweight MLP that takes current state and outputs
    action distribution + value estimate.
    """

    def __init__(self, state_dim: int = 267, hidden_dim: int = 256, n_actions: int = 7):
        super().__init__()

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Actor head: action probabilities
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )

        # Critic head: state value
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Strategy head: which split strategy to use (only when recursing)
        self.strategy_head = nn.Sequential(
            nn.Linear(hidden_dim, len(SPLIT_STRATEGIES)),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.state_encoder(state)
        action_logits = self.action_head(h)
        value = self.value_head(h).squeeze(-1)
        strategy_logits = self.strategy_head(h)
        return action_logits, value, strategy_logits

    def get_action(self, state: torch.Tensor) -> Tuple[Action, torch.Tensor, torch.Tensor]:
        action_logits, value, strategy_logits = self.forward(state)

        action_dist = Categorical(logits=action_logits)
        action_idx = action_dist.sample()
        log_prob = action_dist.log_prob(action_idx)

        strategy_dist = Categorical(logits=strategy_logits)
        strategy_idx = strategy_dist.sample()

        action = Action(
            action_type=action_idx.item(),
            split_strategy=SPLIT_STRATEGIES[strategy_idx.item()],
        )
        return action, log_prob, value


# ─────────────────────────────────────────────
# GRPO Trainer (DeepSeek-R1 inspired)
# ─────────────────────────────────────────────

class GRPOTrainer:
    """
    Group Relative Policy Optimization.

    Key insight from DeepSeek-R1: no critic model needed.
    For each prompt, sample K different recursion strategies,
    compute rewards, then use group-relative advantages.

    R = accuracy - λ * (compute_used / budget)
    """

    def __init__(
        self,
        policy: RecursionPolicyNetwork,
        lr: float = 3e-6,
        kl_coeff: float = 0.001,
        clip_ratio: float = 0.2,
        group_size: int = 16,
        lambda_compute: float = 0.3,
    ):
        self.policy = policy
        self.ref_policy = RecursionPolicyNetwork()  # frozen reference
        self.ref_policy.load_state_dict(policy.state_dict())
        self.ref_policy.eval()

        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
        self.kl_coeff = kl_coeff
        self.clip_ratio = clip_ratio
        self.group_size = group_size
        self.lambda_compute = lambda_compute

    def compute_reward(
        self,
        answer_correct: bool,
        answer_quality: float,  # 0-1 from evaluation metric
        tokens_used: int,
        calls_used: int,
        max_tokens: int,
        max_calls: int,
    ) -> float:
        """
        Composite reward: quality - compute cost.
        Inspired by IBPO and BudgetThinker.
        """
        quality_reward = answer_quality * (2.0 if answer_correct else 0.0)
        compute_penalty = self.lambda_compute * (
            0.5 * tokens_used / max_tokens +
            0.5 * calls_used / max_calls
        )
        # Bonus for solving with fewer resources
        efficiency_bonus = 0.5 * (1 - tokens_used / max_tokens) if answer_correct else 0
        return quality_reward - compute_penalty + efficiency_bonus

    def train_step(
        self,
        prompts: List[dict],
        rlm_engine,  # The RLM engine to run with different strategies
    ) -> dict:
        """
        One GRPO training step.

        For each prompt:
        1. Sample K different recursion strategies
        2. Run RLM with each strategy
        3. Compute rewards
        4. Group-relative advantage estimation (z-score within group)
        5. Policy gradient update
        """
        all_log_probs = []
        all_advantages = []
        all_values = []

        for prompt in prompts:
            group_rewards = []
            group_log_probs = []
            group_values = []

            # Sample K different strategy rollouts
            for _ in range(self.group_size):
                trajectory = self._rollout(prompt, rlm_engine)
                reward = self.compute_reward(
                    trajectory["correct"],
                    trajectory["quality"],
                    trajectory["tokens_used"],
                    trajectory["calls_used"],
                    trajectory["max_tokens"],
                    trajectory["max_calls"],
                )
                group_rewards.append(reward)
                group_log_probs.append(trajectory["total_log_prob"])
                group_values.append(trajectory["total_value"])

            # Group-relative advantage: z-score normalization
            rewards_tensor = torch.tensor(group_rewards)
            advantages = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

            all_advantages.extend(advantages.tolist())
            all_log_probs.extend(group_log_probs)
            all_values.extend(group_values)

        # PPO-style policy gradient with clipping
        log_probs = torch.stack(all_log_probs)
        advantages = torch.tensor(all_advantages)

        with torch.no_grad():
            # Get reference policy log probs for KL penalty
            ref_log_probs = log_probs.detach()  # simplified

        ratio = torch.exp(log_probs - ref_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # KL penalty
        kl_penalty = self.kl_coeff * (log_probs - ref_log_probs).mean()

        total_loss = policy_loss + kl_penalty

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "kl_penalty": kl_penalty.item(),
            "mean_reward": np.mean([r for r in all_advantages]),
        }

    def _rollout(self, prompt: dict, rlm_engine) -> dict:
        """Execute one RLM rollout with policy-guided recursion decisions."""
        # This calls the RLM engine but replaces its fixed recursion logic
        # with decisions from the policy network
        trajectory = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "values": [],
            "tokens_used": 0,
            "calls_used": 0,
            "max_tokens": 50000,
            "max_calls": 20,
        }

        # Run RLM with policy
        result = rlm_engine.run_with_policy(
            prompt["document"],
            prompt["question"],
            policy=self.policy,
            trajectory=trajectory,
        )

        trajectory["correct"] = result["correct"]
        trajectory["quality"] = result["quality_score"]
        trajectory["total_log_prob"] = torch.stack(trajectory["log_probs"]).sum()
        trajectory["total_value"] = torch.stack(trajectory["values"]).mean()
        return trajectory


# ─────────────────────────────────────────────
# Curriculum Training (BudgetThinker-inspired)
# ─────────────────────────────────────────────

class CurriculumScheduler:
    """
    Progressive budget tightening during training.

    Phase 1: Generous budget → learn to solve problems
    Phase 2: Moderate budget → learn to be efficient
    Phase 3: Tight budget → learn to prioritize
    """

    def __init__(self, total_steps: int):
        self.total_steps = total_steps

    def get_budget(self, step: int) -> dict:
        progress = step / self.total_steps

        if progress < 0.3:
            # Phase 1: Learn to solve
            return {"max_depth": 10, "max_calls": 50, "max_tokens": 200000}
        elif progress < 0.7:
            # Phase 2: Learn efficiency
            return {"max_depth": 5, "max_calls": 20, "max_tokens": 100000}
        else:
            # Phase 3: Learn prioritization
            return {"max_depth": 3, "max_calls": 10, "max_tokens": 50000}

    def get_lambda_compute(self, step: int) -> float:
        """Increase compute penalty over training."""
        progress = step / self.total_steps
        return 0.1 + 0.4 * progress  # 0.1 → 0.5
```

### Training Pipeline

```
Phase 1 (SFT Pre-training):
  - Run RLM with various fixed strategies on training set
  - Record (state, optimal_action) pairs from best-performing strategies
  - Pre-train policy via supervised learning on these pairs

Phase 2 (GRPO Fine-tuning):
  - Sample K=16 rollouts per prompt
  - Compute group-relative advantages
  - Update policy with clipped surrogate objective
  - Curriculum: generous → tight budget over training

Phase 3 (Evaluation):
  - Compare learned policy vs fixed heuristics on held-out benchmarks
  - Plot Pareto frontier: quality vs compute at various budget levels
  - Ablate: remove state features one at a time to see what matters
```

### Expected Outcomes

- 20-40% compute reduction at equivalent quality (based on IBPO results)
- Emergent behaviors: model should learn to answer easy questions directly (depth=0) and only recurse deeply for complex multi-hop questions
- Interpretable decisions: can visualize which features drive RECURSE vs ANSWER decisions

---

## Idea 3: RLM for Code — Recursive Code Oracle

### Core Insight

When a senior developer investigates a bug or understands unfamiliar code, they **recursively follow the call chain**: start at the entry point, trace function calls, read relevant definitions, build understanding bottom-up. RLM's recursive decomposition maps perfectly to this workflow. Unlike generic document RLM (which chunks text arbitrarily), code has **natural recursion boundaries** — functions, classes, modules, import chains.

### Literature Foundation

| Paper | Key Technique | Relevance |
|---|---|---|
| **SWE-agent** (NeurIPS 2024, arxiv 2405.15793) | Agent-Computer Interface with `find_file`, `search_file`, `open`, `edit` primitives | Foundational code agent; 18% on SWE-bench Lite; truncated outputs prevent context overflow |
| **AutoCodeRover** (ISSTA 2024, arxiv 2404.05427) | AST-based search primitives: `search_class`, `search_method_in_class` | Structure-aware; 46.2% SWE-bench Verified at <$0.70/issue |
| **Agentless** (FSE 2025, arxiv 2407.01489) | Three-phase hierarchical localization: file → class/function → edit-location | Non-agentic; 32% SWE-bench Lite at $0.34/issue; simpler is competitive |
| **CodexGraph** (NAACL 2025, arxiv 2408.03910) | LLM writes Cypher-like graph queries against code graph DB | Multi-hop reasoning: "find callers of X inheriting from Y"; typed edges (CONTAINS, INHERITS, CALLS, IMPORTS) |
| **Code Graph Model (CGM)** (arxiv 2505.16901) | Graph-aware attention masks restricting message passing to neighboring code graph nodes | 43% SWE-bench Lite (best open-weight); requires custom architecture |
| **Aider Repo Map** | Tree-sitter AST parsing + PageRank-based file ranking | 130+ languages; graph of cross-file references; token-budgeted output |
| **cAST** (2025) | AST-aware chunking (functions, classes as units) | Better retrieval precision than fixed-size windows |
| **SERA** (Ai2, 2026, arxiv 2601.20789) | Soft-verified trajectory generation for training code agents | 54.2% SWE-bench Verified; 26x cheaper than RL |

### Architecture Design

```
┌──────────────────────────────────────────────────────────────┐
│                    RECURSIVE CODE ORACLE                      │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐     ┌──────────────────┐     ┌───────────┐ │
│  │  BUILD-TIME  │     │     RUNTIME      │     │  QUERY    │ │
│  │  INDEXING    │────>│  NAVIGATION API  │<────│  ENGINE   │ │
│  └─────────────┘     └──────────────────┘     └───────────┘ │
│        │                      │                       │      │
│        ▼                      ▼                       ▼      │
│  ┌─────────────┐     ┌──────────────────┐     ┌───────────┐ │
│  │ Tree-sitter  │     │ Prompt Access    │     │ RLM       │ │
│  │ AST Parser   │     │ Primitives       │     │ Recursion │ │
│  │              │     │ (code-aware)     │     │ Engine    │ │
│  │ Symbol Table │     │                  │     │           │ │
│  │ Call Graph   │     │ get_symbol()     │     │ Depth-    │ │
│  │ Import Graph │     │ get_callers()    │     │ guided    │ │
│  │ Type Index   │     │ get_callees()    │     │ traversal │ │
│  │ File Outline │     │ get_outline()    │     │           │ │
│  └─────────────┘     │ search_code()    │     └───────────┘ │
│                       │ get_type_tree()  │                    │
│                       │ get_tests()      │                    │
│                       │ get_changes()    │                    │
│                       └──────────────────┘                    │
│                                                               │
│  RECURSION PATTERN (follows code structure):                  │
│  ─────────────────────────────────────────                    │
│  D0: get_outline(entry_file)                                  │
│      → identifies relevant functions                          │
│  D1: get_symbol(func_name) for each relevant function        │
│      → reads function body, identifies dependencies           │
│  D2: get_callees(func_name) → get_symbol(dependency)         │
│      → follows the call chain into helper functions           │
│  D3: get_callers(helper) if needed                           │
│      → checks if other callers reveal shared patterns         │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Detailed Implementation

```python
import ast
import os
import re
import json
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict

# Requires: pip install tree-sitter tree-sitter-python tree-sitter-javascript
# (or use tree-sitter-languages for multi-language support)


@dataclass
class Symbol:
    """A code symbol (function, class, variable, etc.)."""
    name: str
    kind: str           # "function", "class", "method", "variable", "import"
    file_path: str
    start_line: int
    end_line: int
    signature: str      # function/class signature line
    docstring: Optional[str] = None
    parent_class: Optional[str] = None
    decorators: List[str] = field(default_factory=list)


@dataclass
class CallEdge:
    """An edge in the call graph."""
    caller: str         # fully qualified name
    callee: str         # fully qualified name
    call_site_file: str
    call_site_line: int


@dataclass
class ImportEdge:
    """An import dependency."""
    importing_module: str
    imported_name: str
    imported_from: Optional[str]  # None for bare imports


class CodeIndex:
    """
    Build-time code index using AST parsing.

    Produces:
    - Symbol table: name → [Symbol]
    - Call graph: function → [functions it calls]
    - Reverse call graph: function → [functions that call it]
    - Import graph: module → [modules it imports]
    - Type hierarchy: class → [parent classes, child classes]
    - File outlines: file → [signatures only]
    """

    def __init__(self, repo_path: str, languages: List[str] = None):
        self.repo_path = Path(repo_path)
        self.languages = languages or ["python"]

        # Core indices
        self.symbols: Dict[str, List[Symbol]] = defaultdict(list)
        self.call_graph: Dict[str, List[CallEdge]] = defaultdict(list)
        self.reverse_call_graph: Dict[str, List[CallEdge]] = defaultdict(list)
        self.import_graph: Dict[str, List[ImportEdge]] = defaultdict(list)
        self.type_hierarchy: Dict[str, Dict[str, List[str]]] = {}  # class → {parents, children}
        self.file_outlines: Dict[str, List[str]] = {}
        self.file_contents: Dict[str, str] = {}

    def build(self):
        """Build all indices by parsing all source files."""
        source_files = self._discover_files()
        for file_path in source_files:
            try:
                content = file_path.read_text(encoding='utf-8', errors='replace')
                self.file_contents[str(file_path)] = content

                if file_path.suffix == '.py':
                    self._index_python_file(str(file_path), content)
                # Extensible: add JS, TS, Go, Rust parsers

            except Exception as e:
                print(f"Warning: Failed to index {file_path}: {e}")

        self._build_reverse_call_graph()
        self._build_type_hierarchy()

    def _discover_files(self) -> List[Path]:
        """Find all source files, respecting .gitignore."""
        extensions = {
            "python": [".py"],
            "javascript": [".js", ".jsx", ".ts", ".tsx"],
            "go": [".go"],
            "rust": [".rs"],
            "java": [".java"],
        }
        valid_exts = set()
        for lang in self.languages:
            valid_exts.update(extensions.get(lang, []))

        files = []
        for ext in valid_exts:
            files.extend(self.repo_path.rglob(f"*{ext}"))

        # Filter out common non-source directories
        skip_dirs = {'.git', 'node_modules', '__pycache__', 'venv', '.venv', 'dist', 'build'}
        files = [f for f in files if not any(d in f.parts for d in skip_dirs)]
        return sorted(files)

    def _index_python_file(self, file_path: str, content: str):
        """Parse a Python file and extract symbols, calls, imports."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return

        lines = content.split('\n')
        outline = []

        for node in ast.walk(tree):
            # Functions and methods
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sig = lines[node.lineno - 1].strip()
                docstring = ast.get_docstring(node)
                parent_class = None
                # Check if inside a class
                for potential_parent in ast.walk(tree):
                    if isinstance(potential_parent, ast.ClassDef):
                        for item in potential_parent.body:
                            if item is node:
                                parent_class = potential_parent.name
                                break

                symbol = Symbol(
                    name=node.name,
                    kind="method" if parent_class else "function",
                    file_path=file_path,
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    signature=sig,
                    docstring=docstring,
                    parent_class=parent_class,
                    decorators=[self._decorator_name(d) for d in node.decorator_list],
                )
                fqn = f"{file_path}::{parent_class}.{node.name}" if parent_class else f"{file_path}::{node.name}"
                self.symbols[node.name].append(symbol)
                outline.append(sig)

                # Extract calls within this function
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        callee_name = self._call_name(child)
                        if callee_name:
                            self.call_graph[fqn].append(CallEdge(
                                caller=fqn,
                                callee=callee_name,
                                call_site_file=file_path,
                                call_site_line=child.lineno,
                            ))

            # Classes
            elif isinstance(node, ast.ClassDef):
                sig = lines[node.lineno - 1].strip()
                bases = [self._node_name(b) for b in node.bases]
                symbol = Symbol(
                    name=node.name,
                    kind="class",
                    file_path=file_path,
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    signature=sig,
                    docstring=ast.get_docstring(node),
                    decorators=[self._decorator_name(d) for d in node.decorator_list],
                )
                self.symbols[node.name].append(symbol)
                outline.append(sig)

                # Record type hierarchy
                self.type_hierarchy[node.name] = {
                    "parents": bases,
                    "children": [],  # filled in _build_type_hierarchy
                }

            # Imports
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    self.import_graph[file_path].append(ImportEdge(
                        importing_module=file_path,
                        imported_name=alias.name,
                        imported_from=None,
                    ))
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    self.import_graph[file_path].append(ImportEdge(
                        importing_module=file_path,
                        imported_name=alias.name,
                        imported_from=node.module,
                    ))

        self.file_outlines[file_path] = outline

    def _build_reverse_call_graph(self):
        """Build callee → [callers] index."""
        for caller, edges in self.call_graph.items():
            for edge in edges:
                self.reverse_call_graph[edge.callee].append(edge)

    def _build_type_hierarchy(self):
        """Fill in children for each class."""
        for class_name, info in self.type_hierarchy.items():
            for parent in info["parents"]:
                if parent in self.type_hierarchy:
                    self.type_hierarchy[parent]["children"].append(class_name)

    @staticmethod
    def _call_name(node: ast.Call) -> Optional[str]:
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    @staticmethod
    def _node_name(node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return str(node)

    @staticmethod
    def _decorator_name(node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.Call):
            return CodeIndex._node_name(node.func)
        return ""


class CodeNavigationAPI:
    """
    Runtime API exposed to the LLM inside the RLM sandbox.
    These are the code-aware prompt access primitives.
    """

    def __init__(self, index: CodeIndex):
        self.index = index

    def get_symbol(self, name: str, file_hint: Optional[str] = None) -> str:
        """
        Get full definition of a function/class/variable by name.
        Returns the complete source code of the symbol.
        """
        matches = self.index.symbols.get(name, [])
        if file_hint:
            matches = [m for m in matches if file_hint in m.file_path]
        if not matches:
            return f"Symbol '{name}' not found."

        results = []
        for sym in matches[:3]:  # Limit to 3 matches to prevent context overflow
            content = self.index.file_contents.get(sym.file_path, "")
            lines = content.split('\n')
            definition = '\n'.join(lines[sym.start_line - 1 : sym.end_line])
            results.append(
                f"# {sym.file_path}:{sym.start_line}-{sym.end_line} ({sym.kind})\n"
                f"{definition}"
            )
        return '\n\n'.join(results)

    def get_callers(self, function_name: str) -> str:
        """
        Who calls this function? Returns list of call sites with context.
        """
        edges = self.index.reverse_call_graph.get(function_name, [])
        if not edges:
            return f"No callers found for '{function_name}'."

        results = []
        for edge in edges[:10]:  # Limit results
            content = self.index.file_contents.get(edge.call_site_file, "")
            lines = content.split('\n')
            # Show 2 lines before and after the call site
            start = max(0, edge.call_site_line - 3)
            end = min(len(lines), edge.call_site_line + 2)
            context = '\n'.join(lines[start:end])
            results.append(
                f"Called by: {edge.caller}\n"
                f"  at {edge.call_site_file}:{edge.call_site_line}\n"
                f"  Context:\n{context}"
            )
        return '\n\n'.join(results)

    def get_callees(self, function_name: str) -> str:
        """
        What does this function call? Returns dependency list.
        """
        # Find FQN for this function
        matching_fqns = [
            fqn for fqn in self.index.call_graph
            if fqn.endswith(f"::{function_name}") or fqn.endswith(f".{function_name}")
        ]
        if not matching_fqns:
            return f"No call graph data for '{function_name}'."

        results = []
        for fqn in matching_fqns:
            edges = self.index.call_graph[fqn]
            callees = [f"  → {e.callee} (line {e.call_site_line})" for e in edges]
            results.append(f"{fqn} calls:\n" + '\n'.join(callees))
        return '\n'.join(results)

    def get_file_outline(self, path: str) -> str:
        """
        Return skeleton: class names, function signatures, no bodies.
        Like a table of contents for the file.
        """
        matching_files = [
            f for f in self.index.file_outlines
            if path in f
        ]
        if not matching_files:
            return f"No file matching '{path}' found."

        results = []
        for file_path in matching_files:
            outline = self.index.file_outlines[file_path]
            results.append(f"# {file_path}\n" + '\n'.join(f"  {line}" for line in outline))
        return '\n'.join(results)

    def search_code(self, pattern: str, scope: Optional[str] = None) -> str:
        """
        Regex search across the codebase.
        Optionally scoped to a file/directory.
        """
        results = []
        files = self.index.file_contents.items()
        if scope:
            files = [(f, c) for f, c in files if scope in f]

        for file_path, content in files:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if re.search(pattern, line):
                    results.append(f"{file_path}:{i+1}: {line.strip()}")
                    if len(results) >= 20:  # Limit results
                        results.append("... (truncated)")
                        return '\n'.join(results)
        return '\n'.join(results) if results else f"No matches for '{pattern}'."

    def get_type_hierarchy(self, class_name: str) -> str:
        """Return parent classes, child classes."""
        info = self.index.type_hierarchy.get(class_name)
        if not info:
            return f"Class '{class_name}' not found in type hierarchy."

        parents = ', '.join(info['parents']) or 'None'
        children = ', '.join(info['children']) or 'None'
        return f"Class: {class_name}\n  Parents: {parents}\n  Children: {children}"

    def get_related_tests(self, function_name: str) -> str:
        """Find test files/functions that likely test this symbol."""
        results = []
        for sym_name, syms in self.index.symbols.items():
            for sym in syms:
                if (sym_name.startswith("test_") or sym_name.startswith("Test")) and \
                   function_name.lower() in sym_name.lower():
                    results.append(f"{sym.file_path}:{sym.start_line} - {sym.signature}")

        # Also search for the function name in test files
        for file_path, content in self.index.file_contents.items():
            if 'test' in file_path.lower():
                if function_name in content:
                    # Find specific test functions
                    for line_num, line in enumerate(content.split('\n'), 1):
                        if 'def test' in line and function_name in content[max(0, content.find(line)-500):content.find(line)+500]:
                            results.append(f"{file_path}:{line_num} - {line.strip()}")

        return '\n'.join(results[:10]) if results else f"No tests found for '{function_name}'."

    def get_change_history(self, symbol_name: str, n: int = 5) -> str:
        """Last n git commits that touched files containing this symbol."""
        matching_files = set()
        for sym in self.index.symbols.get(symbol_name, []):
            matching_files.add(sym.file_path)

        if not matching_files:
            return f"No files found for symbol '{symbol_name}'."

        results = []
        for file_path in matching_files:
            try:
                log = subprocess.run(
                    ["git", "log", f"-{n}", "--oneline", "--", file_path],
                    cwd=str(self.index.repo_path),
                    capture_output=True, text=True, timeout=10
                )
                if log.stdout:
                    results.append(f"History for {file_path}:\n{log.stdout}")
            except Exception:
                pass

        return '\n'.join(results) if results else "No git history available."


class RecursiveCodeOracle:
    """
    The main RLM engine specialized for code understanding.
    Recursively explores code following the call graph structure.
    """

    def __init__(self, repo_path: str, model_backend, max_depth: int = 5):
        self.index = CodeIndex(repo_path)
        self.index.build()
        self.api = CodeNavigationAPI(self.index)
        self.backend = model_backend
        self.max_depth = max_depth

    def query(self, question: str) -> str:
        """
        Answer a question about the codebase using recursive exploration.

        Example flow for "Why does upload fail for files > 10MB?":

        Depth 0: get_file_outline("upload") → sees upload_handler, validate_size, process_file
        Depth 1: get_symbol("upload_handler") → reads the function
                 get_symbol("validate_size") → sees size limit is 100MB (not the issue)
        Depth 1: get_symbol("process_file") → sees it calls chunk_and_store
                 get_callees("process_file") → process_file → chunk_and_store → s3_upload
        Depth 2: get_symbol("chunk_and_store") → FOUND: max_parts=2, chunk=5MB → 10MB limit
        Answer: "chunk_and_store limits to 2 parts × 5MB = 10MB at line 47"
        """
        system_prompt = self._build_system_prompt()
        return self._recursive_explore(question, system_prompt, depth=0, context="")

    def _build_system_prompt(self) -> str:
        return """You are a code exploration agent. You have access to these tools:

get_symbol(name)          - Get full source code of a function/class
get_callers(name)         - Find all call sites of a function
get_callees(name)         - List all functions called by a function
get_file_outline(path)    - Get file skeleton (signatures only, no bodies)
search_code(pattern)      - Regex search across codebase
get_type_hierarchy(class) - Get parent/child classes
get_related_tests(name)   - Find test functions for a symbol
get_change_history(name)  - Recent git commits affecting a symbol

Strategy:
1. Start with get_file_outline() to understand structure
2. Use get_symbol() to read specific functions
3. Follow the call chain with get_callees() when you need to understand dependencies
4. Use get_callers() to understand how a function is used
5. When you have enough information, provide your answer

Output your tool calls as Python code. When ready to answer, use: ANSWER: <your answer>"""

    def _recursive_explore(
        self, question: str, system_prompt: str, depth: int, context: str
    ) -> str:
        if depth >= self.max_depth:
            return self._force_answer(question, context)

        prompt = f"{system_prompt}\n\nQuestion: {question}\n\nContext so far:\n{context}\n\nWhat would you like to explore next?"

        response = self.backend.generate(prompt)

        # Parse tool calls from response
        tool_results = self._execute_tool_calls(response)

        if "ANSWER:" in response:
            return response.split("ANSWER:")[1].strip()

        # Recurse with accumulated context
        new_context = context + f"\n\n--- Depth {depth} ---\n{response}\n\nTool results:\n{tool_results}"
        return self._recursive_explore(question, system_prompt, depth + 1, new_context)

    def _execute_tool_calls(self, response: str) -> str:
        """Parse and execute tool calls from LLM response."""
        results = []
        tool_patterns = {
            r'get_symbol\(["\'](\w+)["\']\)': self.api.get_symbol,
            r'get_callers\(["\'](\w+)["\']\)': self.api.get_callers,
            r'get_callees\(["\'](\w+)["\']\)': self.api.get_callees,
            r'get_file_outline\(["\']([^"\']+)["\']\)': self.api.get_file_outline,
            r'search_code\(["\']([^"\']+)["\']\)': self.api.search_code,
            r'get_type_hierarchy\(["\'](\w+)["\']\)': self.api.get_type_hierarchy,
            r'get_related_tests\(["\'](\w+)["\']\)': self.api.get_related_tests,
        }

        for pattern, func in tool_patterns.items():
            for match in re.finditer(pattern, response):
                arg = match.group(1)
                try:
                    result = func(arg)
                    results.append(f"{match.group(0)} →\n{result}")
                except Exception as e:
                    results.append(f"{match.group(0)} → Error: {e}")

        return '\n\n'.join(results) if results else "No tool calls detected."

    def _force_answer(self, question: str, context: str) -> str:
        prompt = f"Based on the following exploration, answer the question.\n\nQuestion: {question}\n\nExploration context:\n{context}\n\nProvide your best answer:"
        return self.backend.generate(prompt)
```

### Benchmark Plan

| Benchmark | Description | Metric |
|---|---|---|
| **SWE-bench Lite** | 300 real GitHub issues | % resolved |
| **RepoBench-R** | Cross-file retrieval | Recall@k |
| **Custom: Bug Localization** | Find root cause of synthetic bugs | Line-level accuracy |
| **Custom: Code QA** | Answer questions about open-source repos | Exact match / BLEU |
| **Custom: Call Chain Tracing** | "What happens when X is called?" | Trace completeness |

### Comparison Baselines

1. **Vanilla RAG**: Embed all functions, retrieve top-k by similarity
2. **SWE-agent style**: Flat tool-use agent (no recursion)
3. **Agentless**: Hierarchical localization without agent loop
4. **RLM Code Oracle**: Our recursive approach

---

## Idea 4: Distilling RLM Traces into Direct Long-Context Models

### Core Insight

RLM is powerful but slow — each query requires multiple sequential LLM calls. What if we could use RLM as a **teacher** to train a student model that answers directly in one forward pass? The key insight is that RLM traces tell us **which parts of the document matter** for each question — this is exactly the attention supervision signal a student needs.

### Literature Foundation

| Paper | Key Technique | Relevance |
|---|---|---|
| **DeepSeek-R1 Distillation** | SFT on 800K CoT traces from 671B → 1.5B-70B students | 32B student retains most of 671B teacher's reasoning on math/code |
| **SeerAttention** (NeurIPS 2025, arxiv 2410.13276) | Learned gate (AttnGate) predicts block-level attention importance | Lightweight gate trained with block-level ground truth from full attention; directly applicable to learning "which blocks matter" |
| **Learning to Focus (LeaF)** (arxiv 2506.07851) | Gradient-based identification of confounding tokens + pruning during distillation | Uses teacher-student comparison to find distracting tokens; prunes them for better attention alignment |
| **HALO + HypeNet** (Jan 2026, arxiv 2601.22156) | Distill Transformer → hybrid attention-RNN; only 2.3B tokens (0.01% pre-training) | State-of-the-art long-context distillation; identifies which layers must keep attention vs convert to RNN |
| **RADLADS** (arxiv 2505.03005) | Convert softmax → linear attention via 3-step distillation | 350-700M tokens sufficient; up to 72B models for <$2,000 |
| **Mamba in the Llama** (NeurIPS 2024) | Replace attention heads with Mamba (linear RNN) layers | 25% attention hybrid matches Transformer; natural length extrapolation to 20x training length |
| **Stratos** (arxiv 2510.15992) | Knowledge injection mode for out-of-domain distillation | Given (question, answer), generates plausible reasoning paths for student training |
| **CoT Structure Analysis** (Snorkel AI) | Structure of CoT matters more than content correctness | Students learn decomposition patterns, not individual step verification |

### Architecture Design

```
┌──────────────────────────────────────────────────────────────┐
│              RLM TRACE DISTILLATION PIPELINE                  │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  PHASE 1: TRACE COLLECTION                                    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  For each (document, question) pair:                    │  │
│  │                                                         │  │
│  │  RLM Recursion ──────► Trace Log:                       │  │
│  │   │                    • accessed_spans: [(3,47),       │  │
│  │   │                      (128,195), (301,340)]          │  │
│  │   │                    • decomposition_tree: {...}       │  │
│  │   │                    • per_step_attention: [...]       │  │
│  │   └── Answer ──────►  • final_answer: "..."             │  │
│  │                        • quality_score: 0.92             │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  PHASE 2: TRAINING DATA GENERATION                            │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  From each trace, generate:                             │  │
│  │                                                         │  │
│  │  1. Linearized reasoning path:                          │  │
│  │     "The answer is in spans X, Y, Z because..."        │  │
│  │                                                         │  │
│  │  2. Attention ground truth (for SeerAttention gate):    │  │
│  │     Block importance mask derived from accessed_spans   │  │
│  │                                                         │  │
│  │  3. Span importance ranking:                            │  │
│  │     Which spans were most useful (from recursion depth)│  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  PHASE 3: STUDENT TRAINING                                    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                                                         │  │
│  │  Stage 1 (SFT): Train on linearized reasoning traces   │  │
│  │  Stage 2 (Attention Alignment): LeaF-style pruning     │  │
│  │  Stage 3 (KD): KL-div on logits (teacher vs student)   │  │
│  │  Stage 4 (Curriculum): easy→hard (shallow→deep traces) │  │
│  │                                                         │  │
│  │  Student Architecture Options:                          │  │
│  │  A) Standard Transformer + SeerAttention gate           │  │
│  │  B) Hybrid: 75% Mamba + 25% attention (HALO-style)     │  │
│  │  C) Linear attention (RADLADS-style conversion)         │  │
│  │                                                         │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  PHASE 4: EVALUATION                                          │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Compare:                                               │  │
│  │  • Quality: Student vs Teacher (RLM) vs Vanilla LLM    │  │
│  │  • Speed: 1 forward pass vs N recursive calls          │  │
│  │  • Attention overlap: student attention vs RLM spans    │  │
│  │  • Length generalization: test at 5x, 10x, 20x train   │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Detailed Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json


# ─────────────────────────────────────────────
# Phase 1: Trace Collection
# ─────────────────────────────────────────────

@dataclass
class RLMTrace:
    """Complete trace from one RLM execution."""
    document: str
    question: str
    answer: str
    quality_score: float            # 0-1 from evaluation metric
    total_tokens_used: int
    total_calls: int
    max_depth_reached: int
    accessed_spans: List[Tuple[int, int]]  # (start_char, end_char) of accessed text
    recursion_tree: dict            # Full tree structure
    per_step_logs: List[dict]       # Each step: {depth, code, result, tokens_in, tokens_out}


class TraceCollector:
    """Wraps the RLM engine to collect detailed traces."""

    def __init__(self, rlm_engine):
        self.engine = rlm_engine
        self.traces: List[RLMTrace] = []

    def collect(self, document: str, question: str, ground_truth: Optional[str] = None) -> RLMTrace:
        """Run RLM and collect the full trace."""
        # Instrument the engine to log every action
        trace_data = {
            "accessed_spans": [],
            "per_step_logs": [],
            "recursion_tree": {},
        }

        # Hook into engine's slice/find/split operations
        original_slice = self.engine.sandbox.prompt_store.slice
        def traced_slice(i, j, **kwargs):
            trace_data["accessed_spans"].append((i, j))
            return original_slice(i, j, **kwargs)
        self.engine.sandbox.prompt_store.slice = traced_slice

        # Run the engine
        result = self.engine.run(document, question)

        trace = RLMTrace(
            document=document,
            question=question,
            answer=result["answer"],
            quality_score=self._evaluate(result["answer"], ground_truth),
            total_tokens_used=result["tokens_used"],
            total_calls=result["calls_used"],
            max_depth_reached=result["max_depth"],
            accessed_spans=trace_data["accessed_spans"],
            recursion_tree=trace_data["recursion_tree"],
            per_step_logs=trace_data["per_step_logs"],
        )
        self.traces.append(trace)

        # Restore original
        self.engine.sandbox.prompt_store.slice = original_slice
        return trace

    def _evaluate(self, prediction: str, ground_truth: Optional[str]) -> float:
        if ground_truth is None:
            return 0.5  # Unknown quality
        # Simple exact match + token overlap
        if prediction.strip().lower() == ground_truth.strip().lower():
            return 1.0
        pred_tokens = set(prediction.lower().split())
        gt_tokens = set(ground_truth.lower().split())
        overlap = len(pred_tokens & gt_tokens) / max(len(gt_tokens), 1)
        return overlap

    def save_traces(self, path: str):
        with open(path, 'w') as f:
            for trace in self.traces:
                f.write(json.dumps({
                    "question": trace.question,
                    "answer": trace.answer,
                    "quality_score": trace.quality_score,
                    "accessed_spans": trace.accessed_spans,
                    "total_tokens": trace.total_tokens_used,
                    "total_calls": trace.total_calls,
                    "max_depth": trace.max_depth_reached,
                }) + '\n')


# ─────────────────────────────────────────────
# Phase 2: Training Data Generation
# ─────────────────────────────────────────────

class TraceToTrainingData:
    """Convert RLM traces into training examples for the student."""

    def __init__(self, tokenizer, block_size: int = 128):
        self.tokenizer = tokenizer
        self.block_size = block_size

    def generate_attention_ground_truth(self, trace: RLMTrace) -> torch.Tensor:
        """
        Convert accessed_spans into block-level importance mask.
        This is the supervision signal for the SeerAttention-style gate.
        """
        doc_tokens = self.tokenizer.encode(trace.document)
        n_blocks = (len(doc_tokens) + self.block_size - 1) // self.block_size

        # Create block importance mask
        importance = torch.zeros(n_blocks)

        for start_char, end_char in trace.accessed_spans:
            # Convert char offsets to token offsets
            start_tok = len(self.tokenizer.encode(trace.document[:start_char]))
            end_tok = len(self.tokenizer.encode(trace.document[:end_char]))

            # Mark blocks as important
            start_block = start_tok // self.block_size
            end_block = min(end_tok // self.block_size + 1, n_blocks)
            importance[start_block:end_block] = 1.0

        return importance

    def generate_linearized_reasoning(self, trace: RLMTrace) -> str:
        """
        Generate a direct reasoning path from the RLM trace.
        The student should learn this linearized form.
        """
        # Extract the key spans that were accessed
        key_excerpts = []
        for start, end in sorted(trace.accessed_spans, key=lambda x: x[0]):
            excerpt = trace.document[start:end]
            key_excerpts.append(f"[Span {start}-{end}]: {excerpt[:200]}...")

        linearized = (
            f"Question: {trace.question}\n\n"
            f"Relevant spans identified:\n" +
            '\n'.join(key_excerpts) +
            f"\n\nReasoning: Based on the above spans, the answer is:\n{trace.answer}"
        )
        return linearized

    def create_training_example(self, trace: RLMTrace) -> dict:
        """Create a single training example."""
        return {
            "input": f"Document: {trace.document}\n\nQuestion: {trace.question}",
            "output": trace.answer,
            "linearized_reasoning": self.generate_linearized_reasoning(trace),
            "attention_gt": self.generate_attention_ground_truth(trace),
            "quality": trace.quality_score,
        }


# ─────────────────────────────────────────────
# Phase 3: Student Model with Learned Attention Gate
# ─────────────────────────────────────────────

class SeerAttentionGate(nn.Module):
    """
    Learned gate that predicts block-level attention importance.
    Inspired by SeerAttention (Microsoft, NeurIPS 2025).

    During inference, this gate predicts which document blocks
    are relevant BEFORE computing full attention, enabling
    sparse attention over only the important blocks.
    """

    def __init__(self, hidden_dim: int, block_size: int = 128, n_heads: int = 32):
        super().__init__()
        self.block_size = block_size
        self.n_heads = n_heads

        # Pool Q and K to block level, then predict importance
        self.q_pool = nn.AdaptiveAvgPool1d(1)
        self.k_pool = nn.AdaptiveAvgPool1d(1)

        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_heads),
            nn.Sigmoid(),
        )

    def forward(
        self,
        query: torch.Tensor,      # [batch, n_heads, q_len, head_dim]
        key: torch.Tensor,         # [batch, n_heads, kv_len, head_dim]
    ) -> torch.Tensor:
        """
        Returns block-level gating scores [batch, n_heads, n_kv_blocks].
        Scores > threshold → compute full attention for that block.
        """
        batch, n_heads, kv_len, head_dim = key.shape
        n_blocks = (kv_len + self.block_size - 1) // self.block_size

        # Pool keys to block level
        # Reshape: [batch, n_heads, n_blocks, block_size, head_dim]
        padded_len = n_blocks * self.block_size
        if kv_len < padded_len:
            key = F.pad(key, (0, 0, 0, padded_len - kv_len))

        key_blocks = key.reshape(batch, n_heads, n_blocks, self.block_size, head_dim)
        key_block_repr = key_blocks.mean(dim=3)  # [batch, n_heads, n_blocks, head_dim]

        # Pool query
        query_repr = query.mean(dim=2, keepdim=True)  # [batch, n_heads, 1, head_dim]
        query_repr = query_repr.expand(-1, -1, n_blocks, -1)

        # Concatenate and predict gate
        combined = torch.cat([query_repr, key_block_repr], dim=-1)
        # [batch, n_heads, n_blocks, 2*head_dim]

        gate_scores = self.gate_network(combined)  # [batch, n_heads, n_blocks, n_heads]
        gate_scores = gate_scores.diagonal(dim1=-2, dim2=-1)  # [batch, n_heads, n_blocks]

        return gate_scores


class DistilledLongContextModel(nn.Module):
    """
    Student model that uses SeerAttention gates to achieve
    RLM-quality answers in a single forward pass.
    """

    def __init__(self, base_model, block_size: int = 128, gate_threshold: float = 0.3):
        super().__init__()
        self.base_model = base_model
        self.block_size = block_size
        self.gate_threshold = gate_threshold

        # Add gates to each attention layer
        hidden_dim = base_model.config.hidden_size
        n_heads = base_model.config.num_attention_heads
        n_layers = base_model.config.num_hidden_layers

        self.gates = nn.ModuleList([
            SeerAttentionGate(hidden_dim // n_heads, block_size, n_heads)
            for _ in range(n_layers)
        ])

    def forward(self, input_ids, attention_mask=None, gate_gt=None):
        """
        Forward pass with gated sparse attention.
        If gate_gt is provided (during training), use it as supervision.
        """
        # This would hook into the base model's attention layers
        # to apply gating. Implementation depends on the specific
        # base model architecture (HuggingFace, custom, etc.)
        pass


# ─────────────────────────────────────────────
# Phase 3: Training Loop
# ─────────────────────────────────────────────

class DistillationTrainer:
    """
    Multi-stage distillation trainer.

    Stage 1 (SFT): Supervised fine-tuning on linearized reasoning traces
    Stage 2 (Attention Alignment): Train gates with RLM span supervision
    Stage 3 (KD): KL-divergence between teacher and student logits
    Stage 4 (Curriculum): easy traces first, hard traces last
    """

    def __init__(
        self,
        student: DistilledLongContextModel,
        traces: List[RLMTrace],
        tokenizer,
        lr: float = 2e-5,
        alpha_sft: float = 1.0,
        alpha_gate: float = 0.5,
        alpha_kd: float = 0.3,
    ):
        self.student = student
        self.traces = traces
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
        self.data_gen = TraceToTrainingData(tokenizer)
        self.alpha_sft = alpha_sft
        self.alpha_gate = alpha_gate
        self.alpha_kd = alpha_kd

    def train_stage1_sft(self, epochs: int = 3):
        """Stage 1: SFT on linearized reasoning traces."""
        # Sort by quality, only use high-quality traces
        good_traces = [t for t in self.traces if t.quality_score > 0.7]

        for epoch in range(epochs):
            for trace in good_traces:
                example = self.data_gen.create_training_example(trace)
                input_ids = self.tokenizer.encode(
                    example["linearized_reasoning"], return_tensors="pt"
                )
                labels = self.tokenizer.encode(trace.answer, return_tensors="pt")

                outputs = self.student(input_ids)
                loss = F.cross_entropy(
                    outputs.logits[:, -labels.shape[1]:, :].reshape(-1, outputs.logits.shape[-1]),
                    labels.reshape(-1),
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def train_stage2_gate(self, epochs: int = 5):
        """Stage 2: Train attention gates with RLM span supervision."""
        for epoch in range(epochs):
            for trace in self.traces:
                example = self.data_gen.create_training_example(trace)
                gate_gt = example["attention_gt"]  # Block-level importance

                input_ids = self.tokenizer.encode(example["input"], return_tensors="pt")

                # Forward pass to get gate predictions
                outputs = self.student(input_ids, gate_gt=gate_gt)

                # Gate loss: BCE between predicted importance and RLM ground truth
                gate_loss = 0
                for layer_idx, gate in enumerate(self.student.gates):
                    gate_pred = gate(outputs.queries[layer_idx], outputs.keys[layer_idx])
                    gate_loss += F.binary_cross_entropy(
                        gate_pred.mean(dim=1),  # Average across heads
                        gate_gt.unsqueeze(0).expand(gate_pred.shape[0], -1),
                    )
                gate_loss /= len(self.student.gates)

                self.optimizer.zero_grad()
                (self.alpha_gate * gate_loss).backward()
                self.optimizer.step()

    def train_stage3_curriculum(self, epochs: int = 3):
        """Stage 4: Curriculum from easy to hard traces."""
        # Sort traces by recursion depth (easy = shallow, hard = deep)
        sorted_traces = sorted(self.traces, key=lambda t: t.max_depth_reached)

        for epoch in range(epochs):
            # Progressive: first epoch uses easy traces, last uses all
            cutoff = int(len(sorted_traces) * (0.3 + 0.7 * epoch / epochs))
            batch = sorted_traces[:cutoff]

            for trace in batch:
                example = self.data_gen.create_training_example(trace)
                # Combined SFT + gate training
                # ... (similar to stages 1 and 2 combined)
```

### Expected Results

Based on the literature:

| Metric | RLM Teacher | Distilled Student | Vanilla LLM |
|---|---|---|---|
| Quality (accuracy) | 90% | **85-92%** (85-95% retention per DeepSeek-R1) | 60-70% |
| Latency | ~10s (N calls) | **~1s** (1 forward pass) | ~1s |
| Attention alignment with RLM spans | 100% (by definition) | **~80%** (SeerAttention accuracy) | ~30% (random-ish) |
| Length generalization | Fixed depth | **20x** training length (Mamba-in-Llama) | 1x |

---

## Idea 5: Multi-Resolution Recursive Attention

### Core Insight

Instead of making RLM's recursion happen at the **application level** (generate code → execute → call LLM again), embed the recursive coarse-to-fine pattern **directly into the attention mechanism**. First scan the document at block-level (coarse), identify hot regions, then compute full attention only on those regions. This gives RLM-like selective access with no recursion overhead.

### Literature Foundation

| Paper | Key Technique | Relevance |
|---|---|---|
| **NSA (Native Sparse Attention)** (DeepSeek, ACL 2025, arxiv 2502.11089) | Three-branch: coarse compression + fine top-k selection + sliding window | Hardware-aligned; end-to-end trainable; Triton implementation available |
| **DHSA** (NeurIPS 2025, arxiv 2510.24606) | Dynamic chunk segmentation + chunk-level scoring + token-level refinement | 10x prefill speedup over FlashAttention-2 at 128K; content-adaptive chunks |
| **Multipole Attention** (NeurIPS 2025, arxiv 2506.13059) | Fast Multipole Method: hierarchical k-means centroids for O(n log n) attention | 4.5x speedup; online clustering update kernel; tested on reasoning models |
| **MRA** (Multi-Resolution Analysis) | Wavelet-inspired multi-scale attention decomposition | O(n log n); only ~10% of coefficients needed; HuggingFace implementation |
| **Twilight** (NeurIPS 2025 Spotlight, arxiv 2502.02770) | Top-p adaptive attention pruning (98% sparsity possible) | 15.4x attention speedup; adapts budget per query difficulty |
| **Block Sparse Flash Attention** (arxiv 2512.07011) | Compute full QK scores, skip V-blocks for low-scoring blocks | Built on FlashAttention; maintains >99% accuracy |
| **HOMER** (ICLR 2024, arxiv 2404.10308) | Training-free divide-and-conquer: chunk → process → merge with token reduction | O(log n) memory; progressive merge at intermediate layers |
| **CCA-Attention** (ICML 2025, arxiv 2412.12465) | Two-branch: globality-aware pooling + locality-preserving attention | Near-linear; 3-6x speedup; drop-in replacement |
| **Squeezed Attention** (ACL 2025, arxiv 2411.09688) | Hierarchical k-means on keys; semantic clustering for selection | 4x+ speedup; semantic > positional grouping |
| **FlexAttention** (PyTorch 2024) | Python-level `score_mod`/`mask_mod` compiled to fused CUDA kernels | Best prototyping tool; 2x faster with block-sparse masks |
| **HDT** (COLM 2024, arxiv 2407.08330) | Hierarchical document structure → anchor tokens at each level | Linear complexity; custom sparse attention kernel for dynamic hierarchies |

### Architecture Design

```
┌──────────────────────────────────────────────────────────────┐
│             MULTI-RESOLUTION RECURSIVE ATTENTION              │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  PASS 1: COARSE SCAN (Block-Level)                            │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Document: [████████████████████████████████████████]   │  │
│  │  Blocks:   [B1] [B2] [B3] [B4] [B5] [B6] [B7] [B8]   │  │
│  │  Pool K:   [k1] [k2] [k3] [k4] [k5] [k6] [k7] [k8]  │  │
│  │  Q·k_i:   [0.1][0.8][0.2][0.9][0.1][0.3][0.7][0.1]   │  │
│  │  Select:        [B2]      [B4]           [B7]          │  │
│  │                                                         │  │
│  │  Cost: O(n/B) — scan 8 blocks instead of 1024 tokens   │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  PASS 2: FINE ATTENTION (Token-Level on selected blocks)      │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Selected: [B2: tok_129..tok_256]                       │  │
│  │            [B4: tok_385..tok_512]                       │  │
│  │            [B7: tok_769..tok_896]                       │  │
│  │                                                         │  │
│  │  Full attention Q × K^T on 384 tokens (not 1024)       │  │
│  │                                                         │  │
│  │  Cost: O(3B × q_len) — 3x cheaper for 3/8 blocks       │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  PASS 3: SLIDING WINDOW (Always-On Local Context)             │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Last W tokens always get full attention                │  │
│  │  Combined with selected blocks via log-sum-exp merge    │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  OPTIONAL PASS 0: SUPER-COARSE (for very long sequences)     │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  For seq > 64K: pool blocks into super-blocks (4×4)    │  │
│  │  Super-scan first, then fine-scan within selected super │  │
│  │  O(n / B^2) for initial scan                           │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  ADAPTIVE BUDGET (Twilight-inspired):                         │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Instead of fixed top-k blocks, use top-p:             │  │
│  │  Sort block scores descending, select until cumulative  │  │
│  │  score >= p * total_score                               │  │
│  │  → Easy queries attend to 1-2 blocks                    │  │
│  │  → Hard queries attend to many blocks                   │  │
│  │  → 80-98% sparsity depending on query                   │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Detailed Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiResolutionAttention(nn.Module):
    """
    Multi-resolution attention that combines:
    - NSA-style coarse compression + fine selection + sliding window
    - Twilight-style adaptive top-p budget
    - Multipole-style optional hierarchical clustering

    Complexity: O(n/B * d + k*B * d + W * d) where:
    - n = sequence length
    - B = block size
    - k = number of selected blocks (adaptive)
    - W = sliding window size
    - d = head dimension
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        head_dim: int,
        block_size: int = 128,
        window_size: int = 256,
        top_p: float = 0.9,
        use_hierarchical: bool = False,
        super_block_factor: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.window_size = window_size
        self.top_p = top_p
        self.use_hierarchical = use_hierarchical
        self.super_block_factor = super_block_factor
        self.scale = head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(hidden_dim, n_heads * head_dim)
        self.k_proj = nn.Linear(hidden_dim, n_heads * head_dim)
        self.v_proj = nn.Linear(hidden_dim, n_heads * head_dim)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden_dim)

        # Block-level key compression (learned pooling)
        self.block_key_compressor = nn.Sequential(
            nn.Linear(head_dim, head_dim),
            nn.GELU(),
            nn.Linear(head_dim, head_dim),
        )

        # Gate for combining coarse and fine branches
        self.branch_gate = nn.Linear(head_dim, 3)  # coarse, fine, window

    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, seq_len, hidden_dim]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        Q = self.q_proj(hidden_states).reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        K = self.k_proj(hidden_states).reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        V = self.v_proj(hidden_states).reshape(batch_size, seq_len, self.n_heads, self.head_dim)

        # Transpose to [batch, n_heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # === BRANCH 1: Coarse Block-Level Scan ===
        coarse_output, selected_blocks = self._coarse_scan(Q, K, V)

        # === BRANCH 2: Fine Token-Level Attention on Selected Blocks ===
        fine_output = self._fine_attention(Q, K, V, selected_blocks)

        # === BRANCH 3: Sliding Window (Always-On Local) ===
        window_output = self._sliding_window(Q, K, V)

        # === Combine Branches ===
        # Learned gating per head per position
        gate_input = Q.mean(dim=-1, keepdim=True)  # Simple: use Q as gate input
        gate_weights = F.softmax(self.branch_gate(Q), dim=-1)  # [batch, heads, seq, 3]

        output = (
            gate_weights[..., 0:1] * coarse_output +
            gate_weights[..., 1:2] * fine_output +
            gate_weights[..., 2:3] * window_output
        )

        # Reshape and project output
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.n_heads * self.head_dim)
        return self.o_proj(output)

    def _coarse_scan(
        self,
        Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Coarse block-level attention scan.
        Pool keys to block representations, compute Q × K_block scores,
        select top-p blocks.
        """
        batch, n_heads, seq_len, head_dim = K.shape
        n_blocks = (seq_len + self.block_size - 1) // self.block_size

        # Pad sequence to multiple of block_size
        pad_len = n_blocks * self.block_size - seq_len
        if pad_len > 0:
            K_padded = F.pad(K, (0, 0, 0, pad_len))
            V_padded = F.pad(V, (0, 0, 0, pad_len))
        else:
            K_padded = K
            V_padded = V

        # Pool keys to block level
        K_blocks = K_padded.reshape(batch, n_heads, n_blocks, self.block_size, head_dim)
        K_block_repr = K_blocks.mean(dim=3)  # [batch, n_heads, n_blocks, head_dim]

        # Optional: learned compression
        K_block_repr = self.block_key_compressor(K_block_repr)

        # Compute block scores: Q_mean × K_block^T
        Q_mean = Q.mean(dim=2, keepdim=True)  # [batch, n_heads, 1, head_dim]
        block_scores = torch.matmul(Q_mean, K_block_repr.transpose(-2, -1)) * self.scale
        block_scores = block_scores.squeeze(2)  # [batch, n_heads, n_blocks]

        # Adaptive top-p selection (Twilight-inspired)
        selected_blocks = self._top_p_select(block_scores)

        # Compute coarse attention using block representatives
        V_blocks = V_padded.reshape(batch, n_heads, n_blocks, self.block_size, head_dim)
        V_block_repr = V_blocks.mean(dim=3)  # [batch, n_heads, n_blocks, head_dim]

        block_attn = F.softmax(block_scores, dim=-1).unsqueeze(-1)
        coarse_output = (block_attn * V_block_repr.unsqueeze(2)).sum(dim=-2)
        # Expand to full sequence length
        coarse_output = coarse_output.expand(-1, -1, seq_len, -1)

        return coarse_output, selected_blocks

    def _top_p_select(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Adaptive top-p block selection.
        Returns boolean mask [batch, n_heads, n_blocks].
        """
        probs = F.softmax(scores, dim=-1)
        sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
        cumulative_probs = sorted_probs.cumsum(dim=-1)

        # Find cutoff: first position where cumulative > top_p
        mask = cumulative_probs <= self.top_p
        # Always keep at least one block
        mask[..., 0] = True

        # Scatter back to original positions
        selected = torch.zeros_like(probs, dtype=torch.bool)
        selected.scatter_(-1, sorted_indices, mask)

        return selected

    def _fine_attention(
        self,
        Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
        selected_blocks: torch.Tensor,  # [batch, n_heads, n_blocks]
    ) -> torch.Tensor:
        """
        Full token-level attention on selected blocks only.
        """
        batch, n_heads, seq_len, head_dim = Q.shape
        n_blocks = selected_blocks.shape[-1]

        # Expand block mask to token mask
        token_mask = selected_blocks.unsqueeze(-1).expand(-1, -1, -1, self.block_size)
        token_mask = token_mask.reshape(batch, n_heads, -1)  # [batch, n_heads, padded_len]
        token_mask = token_mask[:, :, :seq_len]  # Trim to actual seq_len

        # For each head, compute attention only on selected tokens
        # Use masked attention: set non-selected positions to -inf
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Mask out non-selected KV positions
        mask_value = torch.finfo(attn_weights.dtype).min
        attn_weights = attn_weights.masked_fill(~token_mask.unsqueeze(2), mask_value)

        attn_weights = F.softmax(attn_weights, dim=-1)
        fine_output = torch.matmul(attn_weights, V)

        return fine_output

    def _sliding_window(
        self,
        Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sliding window attention for local context.
        Always attend to the last W tokens.
        """
        batch, n_heads, seq_len, head_dim = Q.shape

        # Create causal sliding window mask
        positions = torch.arange(seq_len, device=Q.device)
        # Each position attends to positions within [pos-W, pos]
        row_pos = positions.unsqueeze(1)  # [seq_len, 1]
        col_pos = positions.unsqueeze(0)  # [1, seq_len]

        window_mask = (col_pos <= row_pos) & (col_pos >= row_pos - self.window_size)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        mask_value = torch.finfo(attn_weights.dtype).min
        attn_weights = attn_weights.masked_fill(~window_mask.unsqueeze(0).unsqueeze(0), mask_value)
        attn_weights = F.softmax(attn_weights, dim=-1)

        return torch.matmul(attn_weights, V)


class MultiResolutionTransformerLayer(nn.Module):
    """Drop-in replacement for standard transformer layer."""

    def __init__(self, hidden_dim: int, n_heads: int, head_dim: int, ff_dim: int, **kwargs):
        super().__init__()
        self.attention = MultiResolutionAttention(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            head_dim=head_dim,
            **kwargs,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim),
        )

    def forward(self, x, attention_mask=None):
        # Pre-norm architecture
        x = x + self.attention(self.norm1(x), attention_mask)
        x = x + self.ffn(self.norm2(x))
        return x
```

### Complexity Analysis

| Component | Standard Attention | Multi-Resolution |
|---|---|---|
| Coarse scan | N/A | O(n/B × d) |
| Block selection | N/A | O(n/B × log(n/B)) |
| Fine attention (k blocks) | N/A | O(k × B × q × d) |
| Sliding window | N/A | O(W × q × d) |
| **Total** | **O(n² × d)** | **O(n/B × d + k×B×q×d + W×q×d)** |

For n=32768, B=128, k=32 (top-p), W=256, q=1 (generation):
- Standard: 32768² = ~1.07B operations
- Multi-res: 256 + 4096 + 256 = ~4.6K operations ≈ **230,000x reduction**

### CUDA Kernel Strategy

1. **Prototype with FlexAttention** (PyTorch): Define `mask_mod` that implements the multi-resolution pattern. Get correctness first.
2. **Optimize with Triton**: Port block selection and sparse attention to custom Triton kernels (NSA has reference implementation).
3. **Production with FlashAttention-3 modification**: Fuse coarse scan + selection + fine attention into a single kernel pass using warp specialization.

---

## Idea 6: Cross-Query Caching for RLM Serving

### Core Insight

In production, multiple users ask different questions about the **same document**. Standard RLM recomputes everything from scratch for each query. A smart serving system should:
1. Cache document-level KV caches (reusable across all queries on that document)
2. Cache subcall results (reusable when different queries trigger similar sub-questions)
3. Route requests to maximize cache hits
4. Use semantic matching to detect equivalent subcalls even with different phrasing

### Literature Foundation

| Paper | Key Technique | Relevance |
|---|---|---|
| **SGLang RadixAttention** (arxiv 2312.07104) | Radix tree KV cache management | 85-95% cache hit rate for shared examples; cache-aware scheduling; up to 6.4x throughput |
| **LMCache** (arxiv 2510.09665) | Multi-tier KV cache: GPU → CPU → disk → remote | Up to 15x throughput; chunk-level (not just prefix) reuse; layer-wise pipelining |
| **CacheBlend** (EuroSys 2025 Best Paper) | Non-prefix KV cache reuse with position-aware fusion | 2.2-3.3x TTFT reduction; reuses KV at arbitrary positions |
| **Mooncake** (FAST 2025 Best Paper, arxiv 2407.00079) | Disaggregated prefill/decode with distributed KV pool | 525% throughput in long-context; RDMA-based transfer |
| **Crusoe MemoryAlloy** | Cluster-wide shared KV memory fabric via RDMA | 9.9x TTFT, 5x throughput; KV-aware gateway routing; linear capacity scaling |
| **IC-Cache** (SOSP 2025, arxiv 2501.12689) | Semantic in-context caching + adaptive model routing | 1.4-5.9x throughput; cost-aware routing across model sizes |
| **vLLM APC** | Hash-based block matching for prefix reuse | 7x TTFT reduction on repeated prompts; default since v0.6+ |
| **CachedAttention** | Multi-turn KV cache retention (only prefill new tokens) | Direct KV cache reuse across conversation turns |
| **Orca** (OSDI 2022) | Iteration-level scheduling (continuous batching) | 2-36x throughput; foundation for all modern serving systems |

### Architecture Design

```
┌──────────────────────────────────────────────────────────────────┐
│                    RLM SERVING SYSTEM                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  LAYER 1: REQUEST ROUTER (KV-aware, MemoryAlloy-inspired)        │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Incoming query: (doc_id, question)                        │  │
│  │                                                             │  │
│  │  Router checks:                                             │  │
│  │  1. Is doc_id's KV cache resident on any GPU?              │  │
│  │     → Route to that GPU (maximize cache hit)               │  │
│  │  2. Are similar subcalls cached?                            │  │
│  │     → Route to node with best subcall cache overlap         │  │
│  │  3. Neither → route to least loaded GPU                     │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  LAYER 2: DOCUMENT KV CACHE (LMCache-style multi-tier)           │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  ┌─────────┐     ┌─────────┐     ┌─────────┐             │  │
│  │  │ GPU HBM │────>│ CPU RAM │────>│  NVMe   │             │  │
│  │  │ (hot)   │     │ (warm)  │     │ (cold)  │             │  │
│  │  └─────────┘     └─────────┘     └─────────┘             │  │
│  │                                                             │  │
│  │  Key: (doc_id, chunk_idx, model_layer) → KV tensors        │  │
│  │  Layer-wise pipelining: fetch layer N+1 while computing N  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  LAYER 3: SUBCALL RESULT CACHE (Semantic + Exact)                │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Exact Cache:                                               │  │
│  │    Key: hash(doc_id + chunk_span + question)               │  │
│  │    Value: subcall answer + KV cache                         │  │
│  │                                                             │  │
│  │  Semantic Cache:                                            │  │
│  │    Key: embedding(question) constrained by doc_id           │  │
│  │    Value: subcall answer (if similarity > threshold)       │  │
│  │    Example: "What errors in section 3?" ≈                  │  │
│  │             "List failures in section 3" → cache hit!      │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  LAYER 4: CACHE-AWARE RLM SCHEDULER (SGLang-inspired)            │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Prioritize subcalls by cache benefit:                      │  │
│  │  1. Subcalls with KV cache already on GPU → highest prio   │  │
│  │  2. Subcalls with KV cache on CPU → medium prio            │  │
│  │  3. Subcalls with no cache → lowest prio                   │  │
│  │                                                             │  │
│  │  Batch together: subcalls sharing the same document chunk   │  │
│  │  (amortize KV cache loading across multiple subcalls)      │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  LAYER 5: POSITION-AWARE KV FUSION (CacheBlend)                 │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  When cached KV is from a different prompt position:       │  │
│  │  1. Load cached KV for the document chunk                   │  │
│  │  2. Update positional encoding for new position             │  │
│  │  3. Selectively recompute cross-attention for boundary      │  │
│  │     tokens (between cached chunk and new context)           │  │
│  │  4. Use fused KV cache for attention computation            │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Detailed Implementation

```python
import torch
import hashlib
import time
import threading
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import numpy as np


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class CacheEntry:
    """A cached KV entry for a document chunk."""
    doc_id: str
    chunk_idx: int
    kv_cache: torch.Tensor      # [n_layers, 2, n_heads, chunk_len, head_dim]
    last_accessed: float
    access_count: int = 0
    size_bytes: int = 0
    tier: str = "gpu"            # "gpu", "cpu", "disk"


@dataclass
class SubcallResult:
    """Cached result from a previous subcall."""
    doc_id: str
    chunk_span: Tuple[int, int]  # (start_chunk, end_chunk)
    question: str
    question_embedding: Optional[torch.Tensor]
    answer: str
    kv_cache: Optional[torch.Tensor]
    quality_score: float
    timestamp: float


@dataclass
class ServingRequest:
    """An incoming RLM serving request."""
    request_id: str
    doc_id: str
    question: str
    priority: float = 0.0
    submitted_at: float = 0.0
    cache_benefit: float = 0.0   # Estimated cache hit value


# ─────────────────────────────────────────────
# Document KV Cache (Multi-Tier)
# ─────────────────────────────────────────────

class MultiTierKVCache:
    """
    LMCache-inspired multi-tier storage for document KV caches.

    Tiers: GPU HBM → CPU DRAM → NVMe SSD
    Layer-wise pipelining: overlap fetch and compute.
    """

    def __init__(
        self,
        gpu_budget_gb: float = 8.0,
        cpu_budget_gb: float = 32.0,
        disk_path: str = "/tmp/rlm_kv_cache/",
    ):
        self.gpu_budget = int(gpu_budget_gb * 1024**3)
        self.cpu_budget = int(cpu_budget_gb * 1024**3)
        self.disk_path = disk_path

        self.gpu_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.cpu_cache: OrderedDict[str, CacheEntry] = OrderedDict()

        self.gpu_used = 0
        self.cpu_used = 0

        self._lock = threading.Lock()

    def _make_key(self, doc_id: str, chunk_idx: int) -> str:
        return f"{doc_id}::{chunk_idx}"

    def get(self, doc_id: str, chunk_idx: int) -> Optional[CacheEntry]:
        """Retrieve KV cache, promoting through tiers if needed."""
        key = self._make_key(doc_id, chunk_idx)

        with self._lock:
            # Check GPU first
            if key in self.gpu_cache:
                entry = self.gpu_cache[key]
                entry.last_accessed = time.time()
                entry.access_count += 1
                self.gpu_cache.move_to_end(key)
                return entry

            # Check CPU
            if key in self.cpu_cache:
                entry = self.cpu_cache[key]
                # Promote to GPU
                self._promote_to_gpu(key, entry)
                return entry

        return None  # Cache miss

    def put(
        self, doc_id: str, chunk_idx: int, kv_cache: torch.Tensor
    ) -> CacheEntry:
        """Store KV cache, starting on GPU."""
        key = self._make_key(doc_id, chunk_idx)
        size = kv_cache.element_size() * kv_cache.nelement()

        entry = CacheEntry(
            doc_id=doc_id,
            chunk_idx=chunk_idx,
            kv_cache=kv_cache,
            last_accessed=time.time(),
            size_bytes=size,
            tier="gpu",
        )

        with self._lock:
            self._ensure_gpu_space(size)
            self.gpu_cache[key] = entry
            self.gpu_used += size

        return entry

    def _ensure_gpu_space(self, needed: int):
        """Evict LRU entries from GPU to CPU to make space."""
        while self.gpu_used + needed > self.gpu_budget and self.gpu_cache:
            key, entry = self.gpu_cache.popitem(last=False)  # LRU
            self.gpu_used -= entry.size_bytes
            # Demote to CPU
            entry.kv_cache = entry.kv_cache.cpu()
            entry.tier = "cpu"
            self._ensure_cpu_space(entry.size_bytes)
            self.cpu_cache[key] = entry
            self.cpu_used += entry.size_bytes

    def _ensure_cpu_space(self, needed: int):
        """Evict LRU entries from CPU to disk."""
        while self.cpu_used + needed > self.cpu_budget and self.cpu_cache:
            key, entry = self.cpu_cache.popitem(last=False)
            self.cpu_used -= entry.size_bytes
            # Could save to disk here; for now just evict
            del entry

    def _promote_to_gpu(self, key: str, entry: CacheEntry):
        """Move entry from CPU to GPU."""
        self.cpu_cache.pop(key, None)
        self.cpu_used -= entry.size_bytes

        self._ensure_gpu_space(entry.size_bytes)
        entry.kv_cache = entry.kv_cache.cuda()
        entry.tier = "gpu"
        entry.last_accessed = time.time()
        entry.access_count += 1
        self.gpu_cache[key] = entry
        self.gpu_used += entry.size_bytes

    def get_stats(self) -> dict:
        return {
            "gpu_entries": len(self.gpu_cache),
            "gpu_used_mb": self.gpu_used / (1024**2),
            "cpu_entries": len(self.cpu_cache),
            "cpu_used_mb": self.cpu_used / (1024**2),
        }


# ─────────────────────────────────────────────
# Subcall Result Cache (Exact + Semantic)
# ─────────────────────────────────────────────

class SubcallCache:
    """
    Cache subcall results for reuse across queries.

    Exact cache: hash-based lookup for identical subcalls.
    Semantic cache: embedding-based similarity for equivalent questions.
    """

    def __init__(
        self,
        embedding_model=None,
        semantic_threshold: float = 0.9,
        max_entries: int = 10000,
    ):
        self.embedding_model = embedding_model
        self.semantic_threshold = semantic_threshold
        self.max_entries = max_entries

        # Exact cache: hash → SubcallResult
        self.exact_cache: Dict[str, SubcallResult] = {}

        # Semantic cache: doc_id → [(embedding, SubcallResult)]
        self.semantic_cache: Dict[str, List[Tuple[torch.Tensor, SubcallResult]]] = {}

    def _exact_key(self, doc_id: str, chunk_span: Tuple[int, int], question: str) -> str:
        raw = f"{doc_id}::{chunk_span}::{question}"
        return hashlib.md5(raw.encode()).hexdigest()

    def lookup(
        self, doc_id: str, chunk_span: Tuple[int, int], question: str
    ) -> Optional[SubcallResult]:
        """
        Look up cached subcall result.
        First tries exact match, then semantic similarity.
        """
        # 1. Exact match
        key = self._exact_key(doc_id, chunk_span, question)
        if key in self.exact_cache:
            return self.exact_cache[key]

        # 2. Semantic match (if embedding model available)
        if self.embedding_model and doc_id in self.semantic_cache:
            q_emb = self._embed(question)
            best_score = 0
            best_result = None

            for cached_emb, result in self.semantic_cache[doc_id]:
                # Only match same chunk span
                if result.chunk_span != chunk_span:
                    continue
                similarity = F.cosine_similarity(q_emb, cached_emb, dim=0).item()
                if similarity > best_score:
                    best_score = similarity
                    best_result = result

            if best_score >= self.semantic_threshold:
                return best_result

        return None

    def store(
        self, doc_id: str, chunk_span: Tuple[int, int], question: str,
        answer: str, kv_cache: Optional[torch.Tensor] = None, quality: float = 0.5,
    ):
        """Store a subcall result."""
        # Evict if at capacity
        if len(self.exact_cache) >= self.max_entries:
            # Remove oldest entry
            oldest_key = min(self.exact_cache, key=lambda k: self.exact_cache[k].timestamp)
            del self.exact_cache[oldest_key]

        q_emb = self._embed(question) if self.embedding_model else None

        result = SubcallResult(
            doc_id=doc_id,
            chunk_span=chunk_span,
            question=question,
            question_embedding=q_emb,
            answer=answer,
            kv_cache=kv_cache,
            quality_score=quality,
            timestamp=time.time(),
        )

        # Store in exact cache
        key = self._exact_key(doc_id, chunk_span, question)
        self.exact_cache[key] = result

        # Store in semantic cache
        if q_emb is not None:
            if doc_id not in self.semantic_cache:
                self.semantic_cache[doc_id] = []
            self.semantic_cache[doc_id].append((q_emb, result))

    def _embed(self, text: str) -> torch.Tensor:
        # Use the embedding model to encode the question
        # This is a placeholder; actual implementation depends on model
        with torch.no_grad():
            return self.embedding_model.encode(text, convert_to_tensor=True)

    def get_stats(self) -> dict:
        return {
            "exact_entries": len(self.exact_cache),
            "semantic_docs": len(self.semantic_cache),
            "semantic_entries": sum(len(v) for v in self.semantic_cache.values()),
        }


# ─────────────────────────────────────────────
# Cache-Aware Request Scheduler
# ─────────────────────────────────────────────

class CacheAwareScheduler:
    """
    SGLang-inspired scheduler that prioritizes requests based on
    cache benefit — how much KV cache can be reused.
    """

    def __init__(self, kv_cache: MultiTierKVCache, subcall_cache: SubcallCache):
        self.kv_cache = kv_cache
        self.subcall_cache = subcall_cache
        self.pending_queue: List[ServingRequest] = []
        self._lock = threading.Lock()

    def submit(self, request: ServingRequest):
        """Submit a request and compute its cache benefit score."""
        request.submitted_at = time.time()
        request.cache_benefit = self._estimate_cache_benefit(request)

        with self._lock:
            self.pending_queue.append(request)
            # Sort by cache benefit (highest first), then by submission time
            self.pending_queue.sort(
                key=lambda r: (-r.cache_benefit, r.submitted_at)
            )

    def get_next_batch(self, batch_size: int = 8) -> List[ServingRequest]:
        """
        Get the next batch of requests, grouped by document for cache locality.
        """
        with self._lock:
            if not self.pending_queue:
                return []

            # Group by doc_id, prioritize groups with GPU-resident caches
            doc_groups: Dict[str, List[ServingRequest]] = {}
            for req in self.pending_queue:
                doc_groups.setdefault(req.doc_id, []).append(req)

            # Sort doc groups by total cache benefit
            sorted_groups = sorted(
                doc_groups.items(),
                key=lambda item: sum(r.cache_benefit for r in item[1]),
                reverse=True,
            )

            # Fill batch from highest-benefit groups
            batch = []
            used_docs = set()
            for doc_id, requests in sorted_groups:
                for req in requests:
                    if len(batch) >= batch_size:
                        break
                    batch.append(req)
                    used_docs.add(doc_id)
                if len(batch) >= batch_size:
                    break

            # Remove batched requests from queue
            batch_ids = {r.request_id for r in batch}
            self.pending_queue = [r for r in self.pending_queue if r.request_id not in batch_ids]

            return batch

    def _estimate_cache_benefit(self, request: ServingRequest) -> float:
        """Estimate how much cache can be reused for this request."""
        benefit = 0.0

        # Check if document KV cache is on GPU (highest benefit)
        for chunk_idx in range(100):  # Check first 100 chunks
            entry = self.kv_cache.get(request.doc_id, chunk_idx)
            if entry is None:
                break
            if entry.tier == "gpu":
                benefit += 1.0
            elif entry.tier == "cpu":
                benefit += 0.3

        # Check subcall cache
        # (would need to predict which subcalls this query will trigger)
        # Simple heuristic: if other queries on same doc have cached results, partial benefit
        if request.doc_id in self.subcall_cache.semantic_cache:
            n_cached = len(self.subcall_cache.semantic_cache[request.doc_id])
            benefit += 0.1 * n_cached

        return benefit


# ─────────────────────────────────────────────
# Main RLM Server
# ─────────────────────────────────────────────

class RLMServer:
    """
    Complete RLM serving system with cross-query caching.
    """

    def __init__(
        self,
        rlm_engine,
        gpu_budget_gb: float = 8.0,
        cpu_budget_gb: float = 32.0,
        max_concurrent: int = 8,
    ):
        self.engine = rlm_engine
        self.kv_cache = MultiTierKVCache(gpu_budget_gb, cpu_budget_gb)
        self.subcall_cache = SubcallCache()
        self.scheduler = CacheAwareScheduler(self.kv_cache, self.subcall_cache)
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)

        # Document registry
        self.documents: Dict[str, str] = {}

    def ingest_document(self, doc_id: str, document: str):
        """
        Pre-process a document: chunk it and optionally pre-compute KV caches.
        """
        self.documents[doc_id] = document

        # Chunk the document
        chunks = self._chunk_document(document)

        # Optionally pre-compute KV caches for all chunks
        # (can be done eagerly or lazily on first query)
        for chunk_idx, chunk_text in enumerate(chunks):
            kv = self.engine.compute_kv_cache(chunk_text)
            self.kv_cache.put(doc_id, chunk_idx, kv)

    async def query(self, doc_id: str, question: str) -> dict:
        """Handle a query with caching."""
        request = ServingRequest(
            request_id=hashlib.md5(f"{doc_id}{question}{time.time()}".encode()).hexdigest(),
            doc_id=doc_id,
            question=question,
        )
        self.scheduler.submit(request)

        # Process (in practice, this would be handled by the scheduler loop)
        result = await self._process_query(doc_id, question)

        return {
            "answer": result["answer"],
            "cache_hits": result.get("cache_hits", 0),
            "cache_misses": result.get("cache_misses", 0),
            "tokens_saved": result.get("tokens_saved", 0),
            "latency_ms": result.get("latency_ms", 0),
        }

    async def _process_query(self, doc_id: str, question: str) -> dict:
        """Process a query using the RLM engine with caching."""
        start_time = time.time()
        cache_hits = 0
        cache_misses = 0
        tokens_saved = 0

        # Wrap the engine's recursive call to intercept subcalls
        original_subcall = self.engine._recursive_call

        def cached_subcall(sub_question, depth, text_span=None):
            nonlocal cache_hits, cache_misses, tokens_saved

            if text_span:
                chunk_span = (text_span.start, text_span.end)

                # Check subcall cache
                cached = self.subcall_cache.lookup(doc_id, chunk_span, sub_question)
                if cached:
                    cache_hits += 1
                    tokens_saved += cached.kv_cache.nelement() if cached.kv_cache is not None else 1000
                    return cached.answer

                # Check KV cache
                kv_entry = self.kv_cache.get(doc_id, chunk_span[0])
                if kv_entry:
                    # Use cached KV (skip prefill)
                    result = self.engine._recursive_call_with_kv(
                        sub_question, depth, text_span, kv_entry.kv_cache
                    )
                else:
                    cache_misses += 1
                    result = original_subcall(sub_question, depth, text_span)

                # Cache the result
                self.subcall_cache.store(
                    doc_id, chunk_span, sub_question,
                    result if isinstance(result, str) else result["answer"],
                )
                return result
            else:
                return original_subcall(sub_question, depth, text_span)

        self.engine._recursive_call = cached_subcall
        answer = self.engine.run(self.documents[doc_id], question)
        self.engine._recursive_call = original_subcall

        return {
            "answer": answer,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "tokens_saved": tokens_saved,
            "latency_ms": (time.time() - start_time) * 1000,
        }

    def _chunk_document(self, document: str, chunk_size: int = 2048) -> List[str]:
        """Split document into chunks for KV caching."""
        words = document.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunks.append(' '.join(words[i:i + chunk_size]))
        return chunks

    def get_stats(self) -> dict:
        return {
            "kv_cache": self.kv_cache.get_stats(),
            "subcall_cache": self.subcall_cache.get_stats(),
            "documents": len(self.documents),
        }
```

### Performance Model

Assuming a document with 10 chunks, average 5 subcalls per query:

| Scenario | KV Cache Hits | Subcall Hits | Prefill Savings | Speedup |
|---|---|---|---|---|
| **Query 1 (cold start)** | 0/10 | 0/5 | 0% | 1.0x |
| **Query 2 (same doc, different Q)** | 10/10 | 1/5 | ~80% | ~3x |
| **Query 3 (same doc, similar Q)** | 10/10 | 3/5 | ~90% | ~5x |
| **Query 10 (warm cache)** | 10/10 | 4/5 | ~95% | ~8x |
| **Query 100 (fully warmed)** | 10/10 | 5/5 | ~99% | ~20x |

### Evaluation Plan

1. **Throughput vs # concurrent users** on same document
2. **Cache hit rate** vs query similarity (measured by embedding distance)
3. **Quality impact**: verify cached results maintain accuracy
4. **Memory efficiency**: GPU memory vs served documents
5. **Latency distribution**: P50, P95, P99 across cache-hit and cache-miss scenarios

---

## Recommended Combination Strategy

### For Maximum Project Impact: The "Full Stack" Approach

Combine **3 ideas** into a coherent system that demonstrates depth across multiple dimensions:

```
┌──────────────────────────────────────────────────────────────┐
│                   COMPLETE RLM++ SYSTEM                       │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Application Layer:                                           │
│  ├── Idea 3: Code Oracle (concrete application, demo-able)   │
│  │   └── AST indexing, call graph, code-aware primitives      │
│  │                                                            │
│  Systems Layer:                                               │
│  ├── Idea 1: Hierarchical KV Cache (core systems contrib)     │
│  │   └── Depth-aware compression, tree-structured sharing     │
│  │                                                            │
│  ML Layer:                                                    │
│  └── Idea 2: Learned Recursion Policy (ML contribution)       │
│      └── GRPO training, adaptive depth/budget decisions       │
│                                                               │
│  Together: A code understanding agent that intelligently      │
│  decides how deep to recurse, sharing KV caches efficiently   │
│  across recursive exploration of a codebase.                  │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Alternative "Research Paper" Approach

Focus on **2 ideas** for deeper analysis:

- **Idea 4 (Distillation) + Idea 5 (Multi-Resolution Attention)**: "Can we eliminate RLM's recursive overhead entirely?" — train a direct model using RLM traces, with multi-resolution attention as the mechanism. Publishable framing: "RLM as a Teacher for Efficient Long-Context Reasoning."

### Priority Order for Implementation

| Week | What to Build | Why |
|---|---|---|
| **Week 1** | Core RLM engine + Code Oracle index (Idea 3) | Get the system running end-to-end |
| **Week 2** | Hierarchical KV Cache (Idea 1) | Major systems optimization, measurable speedup |
| **Week 3** | Learned Recursion Policy (Idea 2) | ML contribution, ablation-rich |
| **Week 4** | Cross-Query Serving (Idea 6) OR Distillation (Idea 4) | Production-readiness or novel contribution |
| **Week 5** | Multi-Resolution Attention (Idea 5) | If time permits — hardest, most novel |
| **Week 6** | Evaluation, ablations, write-up | Pareto curves, comparisons, analysis |

---

## References

### KV Cache Management
- [DeFT: Flash Tree-Attention (ICLR 2025)](https://arxiv.org/abs/2404.00242)
- [SGLang / RadixAttention](https://arxiv.org/abs/2312.07104)
- [LMCache](https://arxiv.org/abs/2510.09665)
- [Mooncake (FAST 2025 Best Paper)](https://arxiv.org/abs/2407.00079)
- [CacheBlend (EuroSys 2025 Best Paper)](https://arxiv.org/abs/2405.16444)
- [PyramidKV (TMLR 2025)](https://arxiv.org/abs/2406.02069)
- [SqueezeAttention (ICLR 2025)](https://arxiv.org/abs/2404.04793)
- [H2O: Heavy-Hitter Oracle (NeurIPS 2023)](https://huggingface.co/papers/2306.14048)
- [KVQuant (NeurIPS 2024)](https://arxiv.org/abs/2401.18079)
- [DeepSeek-V2 / MLA](https://arxiv.org/abs/2405.04434)

### RL for Reasoning
- [DeepSeek-R1 / GRPO](https://arxiv.org/abs/2501.12948)
- [Scaling Test-Time Compute](https://arxiv.org/abs/2408.03314)
- [IBPO: Think Smarter not Harder](https://arxiv.org/abs/2501.17974)
- [Mixture-of-Recursions (NeurIPS 2025)](https://arxiv.org/abs/2507.10524)
- [BudgetThinker](https://arxiv.org/abs/2508.17196)
- [TreeRL (ACL 2025)](https://aclanthology.org/2025.acl-long.604/)
- [ReST-MCTS* (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/76ec4dc30e9faaf0e4b6093eaa377218-Paper-Conference.pdf)
- [Budget-Aware Evaluation (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.1112.pdf)

### Code Understanding
- [SWE-agent (NeurIPS 2024)](https://arxiv.org/abs/2405.15793)
- [AutoCodeRover (ISSTA 2024)](https://arxiv.org/abs/2404.05427)
- [Agentless (FSE 2025)](https://arxiv.org/abs/2407.01489)
- [CodexGraph (NAACL 2025)](https://arxiv.org/abs/2408.03910)
- [Code Graph Model](https://arxiv.org/abs/2505.16901)
- [SERA (Ai2, 2026)](https://arxiv.org/abs/2601.20789)
- [Aider Repo Map](https://aider.chat/2023/10/22/repomap.html)

### Distillation
- [SeerAttention (NeurIPS 2025)](https://arxiv.org/abs/2410.13276)
- [Learning to Focus (LeaF)](https://arxiv.org/abs/2506.07851)
- [HALO / HypeNet (2026)](https://arxiv.org/abs/2601.22156)
- [RADLADS](https://arxiv.org/abs/2505.03005)
- [Mamba in the Llama (NeurIPS 2024)](https://arxiv.org/abs/2408.15237)
- [Stratos](https://arxiv.org/abs/2510.15992)

### Multi-Resolution Attention
- [NSA (DeepSeek, ACL 2025)](https://arxiv.org/abs/2502.11089)
- [DHSA (NeurIPS 2025)](https://arxiv.org/abs/2510.24606)
- [Multipole Attention (NeurIPS 2025)](https://arxiv.org/abs/2506.13059)
- [Twilight (NeurIPS 2025 Spotlight)](https://arxiv.org/abs/2502.02770)
- [CCA-Attention (ICML 2025)](https://arxiv.org/abs/2412.12465)
- [FlexAttention (PyTorch)](https://pytorch.org/blog/flexattention/)
- [HOMER (ICLR 2024)](https://arxiv.org/abs/2404.10308)

### LLM Serving
- [Crusoe MemoryAlloy](https://www.crusoe.ai/resources/blog/crusoe-memoryalloy-reinventing-kv-caching-for-cluster-scale-inference)
- [IC-Cache (SOSP 2025)](https://arxiv.org/abs/2501.12689)
- [vLLM / PagedAttention](https://docs.vllm.ai/en/stable/)
- [Orca (OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/yu)
- [RLM Paper](https://arxiv.org/abs/2512.24601)

---

## Idea 7: System Prompt Prefix Caching

### Core Insight

Every RLM call — parent iterations, all child `rlm_query()` subcalls, and `rlm_query_batched()` — sends the same `RLM_SYSTEM_PROMPT` (~800 tokens) as the first message. Without prefix caching, the LLM server recomputes keys/values for those 800 tokens on every single request, wasting compute proportional to the total number of RLM API calls.

**RadixAttention (SGLang, arxiv 2312.07104)** stores KV blocks in a radix tree keyed by token sequence. Any request sharing a prefix reuses those blocks. OpenAI's API (gpt-4o and newer) automatically caches prompts ≥1024 tokens at 50% cost reduction. vLLM and SGLang support it natively.

### What It Buys RLM

For a typical RLM run with 8-way batched subcalls across 3 iterations:
- Total LLM API calls ≈ 25 (parent iterations + child calls)
- System prompt tokens per call ≈ 800
- Without caching: 25 × 800 = 20,000 tokens computed from scratch
- With prefix caching: only 1 × 800 = 800 tokens computed; rest are cache hits
- **Savings: ~96% reduction in system prompt prefill cost**

Additionally, in benchmarks like OOLONG where the same document is chunked and each chunk is independently analyzed with the same system prompt, prefix caching eliminates essentially all repeated system-prompt compute.

### Implementation: `rlm/serving/prompt_prefix_cache.py`

```python
class PromptPrefixCache:
    """
    Tracks and reports system prompt prefix cache hit rate.
    
    For OpenAI: cached_tokens is reported in usage response directly.
    For vLLM/SGLang: prefix caching is automatic when enabled server-side.
    
    This wrapper:
    1. Counts how many tokens are cached vs. freshly computed
    2. Estimates cost/time savings
    3. Exposes metrics for benchmark comparison
    """
```

**Key metric**: `cached_tokens` reported in OpenAI API usage response — directly measurable without model internals.

### Expected Performance

| Scenario | Calls | Cached Tokens | Savings |
|---|---|---|---|
| Single document, 3 iterations | 3 | 2 × 800 | 67% system prompt prefill |
| Batched subcalls (N=8) | 8 | 7 × 800 | 87.5% |
| Full RLM run on OOLONG | ~25 | ~24 × 800 | 96% |

### Literature

| Paper | Contribution |
|---|---|
| **RadixAttention (SGLang)** | Radix tree for KV reuse; up to 6.4x throughput |
| **LMCache (SIGCOMM'24)** | Cross-session prefix persistence; 3-10x delay savings |
| **Mooncake (FAST'25)** | Distributed KV cache pool; RDMA-based prefix transfer |

---

## Idea 8: Speculative Draft Answering

### Core Insight

Not all RLM questions need the full REPL loop. A question like "What year was the treaty signed?" over a 100K-token document might be answerable with a single LLM call that targets the right section — if the section can be found quickly. Running the full REPL (3+ iterations, multiple subcalls) for such questions wastes 5–10x compute.

**Speculative Decoding** (arxiv 2211.17192) uses a cheap draft model to propose tokens and verifies them with the large model. We adapt this idea to the *RLM orchestration level*:

1. **Draft phase**: Send a lightweight "can you answer directly?" prompt with just the first/last 2K chars of context + the question. If the draft model returns a high-confidence direct answer, return it immediately.
2. **Verify phase** (only if draft fails): Run full RLM with REPL.

This is especially effective for S-NIAH benchmarks where the needle is often near a recognizable pattern.

### Algorithm

```
SPECULATIVE_RLM(document, question):
  # Phase 1: Draft (fast path)
  draft_prompt = f"""Answer from this partial context if you can.
  If you need more context, say exactly: NEED_FULL_ANALYSIS
  
  Context preview: {document[:2000]}...{document[-500:]}
  Question: {question}"""
  
  draft_answer = llm_query(draft_prompt)
  
  if "NEED_FULL_ANALYSIS" not in draft_answer:
    return draft_answer  # 1 LLM call total
  
  # Phase 2: Full RLM (slow path)  
  return full_rlm.completion(document, question)
```

### Implementation: `rlm/enhancements/speculative_rlm.py`

```python
class SpeculativeRLM:
    """
    Two-phase RLM: fast draft attempt → full REPL only if needed.
    
    Inspired by speculative decoding but at the orchestration level.
    Tracks draft_hit_rate and token_savings for benchmarking.
    """
    
    def __init__(self, rlm_engine, draft_context_chars=2500, draft_tail_chars=500):
        self.rlm = rlm_engine
        self.draft_context_chars = draft_context_chars
        self.draft_tail_chars = draft_tail_chars
        self._stats = {"draft_hits": 0, "rlm_fallbacks": 0, "tokens_saved": 0}
    
    def completion(self, document: str, question: str) -> RLMChatCompletion:
        ...
    
    def get_stats(self) -> dict:
        ...
```

### Expected Performance

| Question type | Draft hit rate | Token savings |
|---|---|---|
| Simple fact lookup (S-NIAH) | ~60% | ~85% |
| Multi-hop reasoning (BrowseComp) | ~10% | ~15% |
| Aggregation (OOLONG) | ~5% | ~7% |

### Literature

| Paper | Contribution |
|---|---|
| **SpecDecoding (Leviathan et al., 2023)** | Draft + verify; 2-3x decode speedup |
| **EAGLE (Li et al., 2024)** | Feature-level draft; higher acceptance rate |
| **Adaptive Computation (Graves 2016)** | Skip compute for easy inputs |

---

## Idea 9: Adaptive Window Batching

### Core Insight

The current `AsyncSubcallManager` dispatches subcalls immediately as they are submitted. In a typical `rlm_query_batched([q1, q2, ..., q8])` call, all 8 subcalls are fired at once — but they arrive at the server in rapid succession rather than as a true batch, potentially hitting the server before the previous request's KV cache is warm.

**Continuous Batching (ORCA, SOSP 2023)** groups requests at iteration granularity: as soon as one sequence finishes decoding a token, the next waiting request joins the batch. At the RLM client level, we can implement a complementary optimization: accumulate subcalls in a time window (50–100ms), then dispatch as a coordinated batch.

This improves GPU utilization because:
1. The server sees N requests arrive simultaneously, enabling efficient batch matrix multiplication
2. Requests with shared prefixes (same system prompt) are more likely to be scheduled together, improving RadixAttention hit rates
3. Avoids the "thundering herd" where 8 requests each wait for server-side batch scheduling

### Implementation: Enhance `AsyncSubcallManager` in `rlm/enhancements/async_subcalls.py`

Add `WindowedBatcher` to accumulate calls within a time window before dispatching:

```python
class WindowedBatcher:
    """
    Accumulates calls for `window_ms` milliseconds then dispatches as a batch.
    
    For rlm_query_batched(): calls are always co-dispatched (window = 0).
    For sequential rlm_query() calls: window allows co-arrival at server.
    
    Inspired by ORCA's iteration-level scheduling: fine-grained batching
    reduces head-of-line blocking and improves GPU utilization.
    """
    
    def __init__(self, dispatch_fn, window_ms=50, max_batch_size=32):
        self.dispatch_fn = dispatch_fn
        self.window_ms = window_ms
        self.max_batch_size = max_batch_size
        self._pending = []
        self._lock = threading.Lock()
        self._timer = None
    
    def submit(self, call_args) -> Future:
        ...
    
    def _dispatch_batch(self):
        ...
```

### Serving-Side Integration

When using vLLM or SGLang as backend:
- Enable `--enable-prefix-caching` on the server
- Use `--max-num-seqs 256` to allow large batch sizes
- Set scheduling policy to `fcfs` or `priority` for RLM's recursive calls

### Expected Performance

| Scenario | Without windowing | With windowing (50ms) |
|---|---|---|
| 8 sequential rlm_query() calls | 8 separate server requests | 1–3 batched requests |
| GPU utilization | 40–60% (due to request gaps) | 75–90% |
| Wall clock time (8 calls × 2s each) | 16s sequential or 2s parallel | 1.8s (better batch scheduling) |

### Combined System Stack

Putting all ideas together, the optimal RLM serving stack is:

```
User Request
     │
     ▼
SpeculativeRLM (Idea 8) ◄── Skip REPL for easy questions
     │ (if hard question)
     ▼
EnhancedRLM
  ├── WindowedBatcher (Idea 9) ◄── Batch concurrent subcalls
  ├── SubcallCache ◄── Skip repeated subcalls (Idea 6)
  ├── PromptPrefixCache (Idea 7) ◄── Track cached tokens
  └── HierarchicalKVCache (Idea 1) ◄── Compress ancestor KVs
     │
     ▼
LLM Server (vLLM / SGLang)
  ├── RadixAttention ◄── Prefix KV reuse
  ├── PagedAttention ◄── Variable-length memory
  ├── Continuous Batching ◄── High GPU utilization
  └── INT8 KV Cache ◄── 2x more concurrent requests
```

### Literature

| Paper | Contribution |
|---|---|
| **ORCA (Yu et al., SOSP 2023)** | Iteration-level scheduling, continuous batching |
| **vLLM (Kwon et al., SOSP 2023)** | PagedAttention, high-throughput serving |
| **SGLang (Zheng et al., ICML 2024)** | RadixAttention, cache-aware scheduling |
| **Dynamo (NVIDIA, 2025)** | Disaggregated prefill/decode at scale |

---

## Recommended Combination Strategy (Updated)

| Idea | Effort | Measurable in benchmarks | Speedup |
|---|---|---|---|
| 1. Hierarchical KV Cache | High | Yes (HF models) | 2-8x memory |
| 2. Learned Recursion Policy | High | Yes (quality) | +5-15% accuracy |
| 3. Code Oracle | Medium | LongBench-v2 | +20-30% code QA |
| 4. Distillation | High | Yes (new model) | 3-5x faster |
| 5. Multi-Resolution Attention | Medium | Yes (token savings) | 2-3x token reduction |
| 6. Subcall Cache | Low | Yes (cache hit rate) | 2-10x for repeated queries |
| **7. Prefix Cache** | **Low** | **Yes (cached_tokens)** | **67-96% prefill savings** |
| **8. Speculative Draft** | **Low** | **Yes (draft_hit_rate)** | **1-5x for easy questions** |
| **9. Adaptive Batching** | **Medium** | **Yes (GPU util proxy)** | **1.3-1.8x throughput** |

**For a course project demonstrating systems thinking, combine Ideas 6 + 7 + 8 as they have the highest benefit-to-effort ratio and produce clear, measurable metrics in the paper's benchmarks (S-NIAH, OOLONG).**
