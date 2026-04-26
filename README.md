# CMU 11-868 LLM Systems — Final Project
## Enhanced Recursive Language Models (ERLM)

**Team:** Bhuvan Nallamoto  
**Course:** 11-868 Large Language Model Systems, Carnegie Mellon University  
**Model:** Qwen3-8B (local via Ollama) / Qwen3-Coder-480B-A35B (Vertex AI MaaS)  
**Dataset:** LongBench-v2 CodeQA (long-context multiple-choice QA, docs 500K–5M chars)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [What is RLM?](#3-what-is-rlm)
4. [What is ERLM?](#4-what-is-erlm)
5. [Optimizations in Detail](#5-optimizations-in-detail)
   - [O1 — TF-IDF Prompt Indexer](#o1--tf-idf-prompt-indexer)
   - [O2 — Adaptive Budget Controller](#o2--adaptive-budget-controller)
   - [O3 — Async Parallel Subcall Manager](#o3--async-parallel-subcall-manager)
   - [O4 — KV-Cache Prefix Sharing](#o4--kv-cache-prefix-sharing)
   - [O5 — FP8/INT8 Quantization](#o5--fp8int8-quantization)
6. [Evaluation Design](#6-evaluation-design)
7. [Metrics Explained](#7-metrics-explained)
8. [How to Run](#8-how-to-run)
9. [Expected Results](#9-expected-results)

---

## 1. Project Overview

**Research question:** Can systems-level optimizations applied to a Recursive Language Model (RLM) reduce token usage and latency on long-context QA tasks without sacrificing answer quality?

The core challenge: LongBench-v2 documents are 500K–5M characters long. A Qwen3-8B model has a 40K token context window (~160K characters). No single LLM call can read the full document. The RLM system works around this by running the model in an iterative loop with a Python REPL — but this loop is expensive. ERLM adds optimizations on top to make the loop cheaper and faster.

**Hypothesis:** ERLM's O1+O2+O3 optimizations reduce token consumption by ≥40% and wall-clock time by ≥30% compared to plain RLM baseline, while maintaining comparable exact-match accuracy on CodeQA.

---

## 2. Repository Structure

```
LLMSYS_FINAL_PROJECT/
├── RLM/
│   ├── BASELINE/                   # Original RLM implementation (unmodified)
│   │   └── rlm/
│   │       ├── core/
│   │       │   ├── rlm.py          # Main RLM loop
│   │       │   ├── lm_handler.py   # Manages LM calls, batching, token tracking
│   │       │   └── types.py        # RLMChatCompletion, UsageSummary, etc.
│   │       ├── clients/
│   │       │   ├── openai.py       # OpenAI-compatible client
│   │       │   └── __init__.py     # get_client() router
│   │       ├── environments/
│   │       │   └── base_env.py     # Python REPL environment
│   │       └── utils/
│   │           ├── prompts.py      # System prompt templates
│   │           └── parsing.py      # find_final_answer() extractor
│   │
│   └── ERLM/                       # Our enhanced implementation
│       ├── erlm.py                 # EnhancedRLM — subclass of RLM, wires O1–O5
│       ├── clients/
│       │   ├── vertex_ai.py        # Vertex AI MaaS client (Qwen3-480B)
│       │   └── ollama.py           # Local Ollama client (Qwen3-8B, think:false)
│       ├── optimisations/
│       │   ├── prompt_indexer.py   # O1: TF-IDF indexer + search_context() tool
│       │   ├── budget_controller.py # O2: Adaptive early termination
│       │   ├── async_subcall.py    # O3: Parallel batching instructions + stats
│       │   ├── kv_prefix_cache.py  # O4: vLLM prefix caching client
│       │   └── fp8_quantization.py # O5: FP8/INT8 quantization config
│       └── EVALS/
│           ├── compare.py          # Main evaluation harness
│           ├── benchmarks/
│           │   ├── longbench_codeqa.py  # LongBench-v2 CodeQA loader
│           │   └── oolong.py            # OOLONG dataset loader
│           └── results/            # Auto-saved .log and .jsonl per run
```

---

## 3. What is RLM?

A **Recursive Language Model** is a system where an LLM does not just generate one response — it runs in a loop, using a Python REPL at each step.

### How one iteration works

```
┌─────────────────────────────────────────────┐
│  LLM receives: system prompt + message history │
│  LLM outputs:  reasoning + Python code block  │
└───────────────────┬─────────────────────────┘
                    │ code executed in REPL
                    ▼
┌─────────────────────────────────────────────┐
│  REPL result appended to message history    │
│  Loop repeats (up to max_iterations=5)      │
└───────────────────┬─────────────────────────┘
                    │ if FINAL_ANSWER found
                    ▼
              Return answer
```

### Why this matters for long documents

The document is written to disk and loaded as a Python variable `context` in the REPL. The model never sees 5M characters in its prompt — instead it writes code to read, search, or process chunks of the document iteratively. This sidesteps the context window limit entirely.

### The cost

Each iteration = at least one LLM API call. With `max_iterations=5` and sub-calls (e.g. `llm_query()` inside REPL code), a single sample can consume 5–20 LLM calls and 10K–50K tokens.

---

## 4. What is ERLM?

`EnhancedRLM` is a subclass of `RLM` (no modifications to BASELINE) that adds five optional optimizations via constructor flags:

```python
from erlm import EnhancedRLM

model = EnhancedRLM(
    backend="ollama",
    backend_kwargs={"model_name": "qwen3:8b", "base_url": "http://localhost:11434/v1"},
    max_iterations=5,
    max_timeout=600.0,
    enable_indexing=True,   # O1
    enable_budget=True,     # O2
    enable_async=True,      # O3
    enable_kv_cache=False,  # O4 (requires vLLM server)
    enable_fp8=False,       # O5 (requires NVIDIA GPU)
)

result = model.completion(document, root_prompt=question)
```

All flags default to `False` — with no flags set, ERLM behaves identically to the baseline RLM.

### The two ERLM configurations we evaluate

| Config | Flags | Purpose |
|---|---|---|
| `erlm_o1o2` | O1 + O2 | Core efficiency gains: smarter retrieval + budget enforcement |
| `erlm_o1o2o3` | O1 + O2 + O3 | Adds parallel sub-query execution on top |

We test them separately to measure each optimization's independent contribution.

---

## 5. Optimizations in Detail

### O1 — TF-IDF Prompt Indexer

**File:** `ERLM/optimisations/prompt_indexer.py`  
**Flag:** `enable_indexing=True`

#### Problem
Without O1, the model's REPL code typically reads the document linearly from the start. For a 5M-char document with a 40K-token context window, it sees only the first ~160K characters — missing 97% of the document.

#### Solution
Before the RLM loop starts, O1:
1. **Chunks** the document into 2,000-char pieces with 200-char overlap
2. **Builds a TF-IDF matrix** over all chunks using `sklearn.TfidfVectorizer`
3. **Injects a `search_context(query, top_k=5)` tool** into the REPL environment
4. **Prepends a system prompt note** telling the model to use this tool instead of linear reading

#### During the loop
```python
# Model writes this in its REPL instead of reading doc linearly:
results = search_context("SC-FDMA uplink transmission scheme")
# Returns top-5 most relevant 2000-char chunks from anywhere in the 5M-char doc
```

#### What it measures
- **`o1_chunks`** — number of TF-IDF chunks built (e.g. 2,796 for 5M chars). Must be >0 to confirm O1 fired.
- **`input_tokens` reduction** — targeted chunk retrieval sends less text to the LLM per call than naive linear reading

#### Expected impact
- `input_tokens` for `erlm_o1o2` < `input_tokens` for `rlm_baseline`
- Better chance of finding the relevant passage → EM improvement on hard samples

---

### O2 — Adaptive Budget Controller

**File:** `ERLM/optimisations/budget_controller.py`  
**Flag:** `enable_budget=True`

#### Problem
The baseline RLM runs up to `max_iterations=5` regardless of whether the model is making progress. On hard questions, the model can spend 3-4 iterations repeating the same search with slightly different wording, generating thousands of wasted output tokens.

#### Solution
After each iteration, O2 tracks:
1. **Productivity score** — cosine similarity between consecutive responses (low similarity = model is changing approach = productive; high similarity = model is repeating itself = unproductive)
2. **Rolling productivity average** over the last 3 iterations
3. **Token budget consumption** — how much of `max_tokens` has been used

**Termination rules:**
| Condition | Action |
|---|---|
| Rolling productivity < 30% | `early_stop: low_productivity` |
| Token budget > 75% used | `early_stop: low_budget` (warning mode) |
| Token budget > 90% used | `early_stop: critical_budget` (force exit) |

When termination fires, it patches `iteration.final_answer` with the best partial answer seen so far, causing the RLM loop to exit cleanly on the next check.

#### What it measures
- **`early_terminated`** — True/False per sample
- **`termination_reason`** — which condition fired
- **`O2FireRate`** — % of samples where O2 triggered. Healthy range: 10–60%. 0% = never helped. 100% = too aggressive.
- **`output_tokens` reduction** — fewer iterations = less generation

---

### O3 — Async Parallel Subcall Manager

**File:** `ERLM/optimisations/async_subcall.py`  
**Flag:** `enable_async=True`

#### Problem
When the model needs to process multiple document chunks (e.g., summarize 5 different sections), it naturally writes sequential code:
```python
s1 = llm_query("Summarize section 1")   # 30s
s2 = llm_query("Summarize section 2")   # 30s  → 150s total
s3 = llm_query("Summarize section 3")   # 30s
```

#### Solution
O3 adds a large instruction block to the system prompt:

```
PERFORMANCE CRITICAL — PARALLEL EXECUTION INSTRUCTIONS
Use llm_query_batched([prompt1, prompt2, prompt3]) instead of looping.
This runs ALL prompts in PARALLEL — dramatically faster than sequential calls.
```

The `LMHandler` already implements `llm_query_batched` via `asyncio.gather`. When the model follows the instruction:
```python
summaries = llm_query_batched([
    "Summarize section 1",
    "Summarize section 2",   # All 3 run simultaneously → ~30s total
    "Summarize section 3",
])
```

#### What it measures
- **`o3_parallel_batches`** — number of `llm_query_batched` calls the model actually made. Must be >0 to confirm O3 was utilized.
- **`O3Utilized`** — % of samples where model issued at least one batched call
- **`SpeedupVsBase`** — wall-clock time reduction vs baseline

#### Important caveat
O3's effectiveness depends entirely on whether Qwen3-8B follows the batching instruction. If the model ignores it and continues calling `llm_query()` sequentially, `o3_parallel_batches = 0` and no speedup occurs. This is one of the things the evaluation explicitly measures.

---

### O4 — KV-Cache Prefix Sharing

**File:** `ERLM/optimisations/kv_prefix_cache.py`  
**Flag:** `enable_kv_cache=True`  
**Requirement:** Running vLLM server with `--enable-prefix-caching`

#### Problem
Every RLM iteration re-sends the system prompt + full message history to the LLM. The system prompt (~2K tokens) and early iterations' context are identical across calls — but the LLM recomputes attention for them every time.

#### Solution
vLLM's prefix caching stores the KV (key-value) attention states for common prefixes in GPU memory. When the system prompt tokens appear again in the next call, vLLM reuses the cached states — skipping the prefill computation for those tokens entirely.

`VLLMPrefixCachedClient` wraps `OpenAIClient` pointed at a local vLLM server:
```bash
vllm serve Qwen/Qwen3-8B --enable-prefix-caching --port 8001
```

**Note:** O4 requires an NVIDIA GPU and is not evaluated in the current Ollama-based setup. It's implemented and available for GPU cluster runs.

---

### O5 — FP8/INT8 Quantization

**File:** `ERLM/optimisations/fp8_quantization.py`  
**Flag:** `enable_fp8=True` (requires `enable_kv_cache=True`)  
**Requirement:** NVIDIA A100/H100 GPU

#### Problem
Qwen3-8B in BF16 requires ~16GB of GPU memory. This leaves little room for KV cache and limits batch size, reducing throughput.

#### Solution
O5 detects GPU capabilities and recommends the best quantization:
- **A100/H100:** FP8 weight quantization via `vllm --quantization fp8` — ~2x memory reduction with minimal accuracy loss
- **Other NVIDIA:** INT8 (bitsandbytes) — ~2x memory reduction, slightly higher accuracy loss
- **No GPU / Mac:** No-op (quantization skipped)

**Note:** O5 is implemented but not evaluated in the current Ollama setup. Relevant for GPU cluster deployments.

---

## 6. Evaluation Design

### Dataset
**LongBench-v2 CodeQA** — 500 long-context multiple-choice questions. Each sample has:
- A document: 500K–5M characters (code repositories, papers, technical docs)
- A question: natural language
- Gold answer: single letter A / B / C / D

We filter to documents ≥512K characters to stress-test the long-context handling.

### Methods compared

| Method | Model | Optimizations |
|---|---|---|
| `base_model` | Qwen3-8B | No RLM — direct LLM call, doc truncated to 140K chars |
| `rlm_baseline` | Qwen3-8B | Plain RLM, no ERLM optimizations |
| `erlm_o1o2` | Qwen3-8B | O1 (TF-IDF) + O2 (Budget) |
| `erlm_o1o2o3` | Qwen3-8B | O1 + O2 + O3 (Async) |
| `rlm_finetuned` | rlm-qwen3-8b-v0.1 | Plain RLM, fine-tuned model |
| `erlm_finetuned` | rlm-qwen3-8b-v0.1 | O1 + O2, fine-tuned model |

The `base_model` establishes whether RLM adds any value over a plain LLM call.  
The `rlm_finetuned` / `erlm_finetuned` methods test whether fine-tuning on RLM trajectories compounds with system optimizations.

### Controlled variables
- Same 5 document samples for all methods
- Same `max_iterations=5`, `max_timeout=600s`
- Same Ollama endpoint, same hardware (Apple M4 Pro, 24GB)
- Thinking mode disabled via `think: false` (Ollama) / `enable_thinking: false` (Vertex)

---

## 7. Metrics Explained

### Per-sample (saved to `.jsonl`)

| Metric | Description |
|---|---|
| `exact_match` | 1.0 if predicted letter matches gold letter, else 0.0 |
| `f1` | Token-level F1 (same as EM for single-letter MC answers) |
| `tokens_used` | Total tokens: `input_tokens + output_tokens` |
| `input_tokens` | Prompt tokens sent to LLM (what O1 reduces via targeted retrieval) |
| `output_tokens` | Generated tokens (what O2 reduces via early termination) |
| `tokens_per_sec` | Throughput: `tokens_used / wall_clock_s` |
| `iterations` | RLM loop iterations completed |
| `wall_clock_s` | End-to-end seconds |
| `o1_chunks` | TF-IDF chunks built. 0 = O1 not active. ~2796 for 5M-char doc |
| `early_terminated` | True if O2 budget controller fired |
| `termination_reason` | `"low_productivity"` / `"low_budget"` / `"critical_budget"` / `""` |
| `o3_parallel_batches` | Number of `llm_query_batched` calls model issued. 0 = O3 not utilized |

### Summary-level (printed at end of run)

**Table 1 — Quality & Token Efficiency**

| Column | What it shows |
|---|---|
| EM | Average exact match across samples |
| TotalTok | Average total tokens per sample |
| InTok | Average input tokens (prompt size) |
| OutTok | Average output tokens (generation length) |
| Tok/s | Average throughput in tokens/second |
| Time(s) | Average wall-clock seconds |

**Table 2 — Optimization Verification**

| Column | What it proves |
|---|---|
| `TokRedux%` | `(baseline_tokens - method_tokens) / baseline_tokens × 100`. Positive = tokens saved vs baseline |
| `SpeedupVsBase` | `baseline_time / method_time`. >1.0x = method is faster |
| `TokEff` | `EM / (tokens_used / 1000)`. Correct answers per 1K tokens — higher = more efficient |
| `O1Chunks` | Confirms O1 built an index (>0) |
| `O2FireRate` | % samples where budget controller triggered early stop |
| `O3Utilized` | % samples where model actually called `llm_query_batched` |

---

## 8. How to Run

### Prerequisites
```bash
# Install Ollama (Mac)
brew install ollama

# Pull Qwen3-8B
ollama pull qwen3:8b

# Pull fine-tuned RLM model (optional)
ollama pull hf.co/cameronbergh/rlm-qwen3-8b-v0.1-gguf:Q8_0

# Install Python dependencies
pip install scikit-learn openai google-auth datasets
```

### Sanity check (3 samples, core methods, ~30-45 min)
```bash
cd /Users/bhuvan/Desktop/LLMSYS_FINAL_PROJECT/RLM
python ERLM/EVALS/compare.py \
  --backend ollama \
  --ollama-model qwen3:8b \
  --n 3 \
  --methods rlm_baseline erlm_o1o2 erlm_o1o2o3 \
  --out ERLM/EVALS/results/
```

### Full evaluation (50 samples, all 4 core methods, ~8-12 hrs)
```bash
python ERLM/EVALS/compare.py \
  --backend ollama \
  --n 50 \
  --methods base_model rlm_baseline erlm_o1o2 erlm_o1o2o3 \
  --out ERLM/EVALS/results/
```

### With fine-tuned model (all 6 methods)
```bash
python ERLM/EVALS/compare.py \
  --backend ollama \
  --n 50 \
  --methods base_model rlm_baseline erlm_o1o2 erlm_o1o2o3 rlm_finetuned erlm_finetuned \
  --out ERLM/EVALS/results/
```

### Via Vertex AI (Qwen3-480B, requires GCP auth + quota)
```bash
gcloud auth application-default login
export VERTEX_PROJECT_ID=your-project-id

python ERLM/EVALS/compare.py \
  --backend vertex \
  --project your-project-id \
  --n 5 \
  --out ERLM/EVALS/results/
```

### Output files
Every run produces two files in `--out` directory:
- `compare_YYYYMMDD_HHMMSS.log` — human-readable log with all predictions, token counts, and summary tables
- `compare_YYYYMMDD_HHMMSS.jsonl` — machine-readable, one JSON per sample, written after each sample (crash-safe)

---

## 9. Expected Results

Based on the optimization design, expected outcomes on CodeQA (Qwen3-8B, n=50):

```
Method          TotalTok   InTok   OutTok   Time(s)  TokRedux%  Speedup  O1Chunks  O2Fire%  O3Util%
base_model         ~4,500   ~4,100     ~400    ~30s       —          —          0        0%       0%
rlm_baseline      ~12,000   ~9,500   ~2,500   ~100s      0%         1.0x        0        0%       0%
erlm_o1o2          ~6,000   ~4,000   ~2,000    ~60s     ~50%       ~1.7x     2796      ~30%       0%
erlm_o1o2o3        ~6,000   ~4,000   ~2,000    ~40s     ~50%       ~2.5x     2796      ~30%     ~50%
```

**Key claims:**
1. `base_model` is fastest and cheapest but misses 97% of the document — expected low EM
2. `rlm_baseline` iteratively explores but wastefully — high token cost, moderate EM
3. `erlm_o1o2` finds relevant chunks faster (O1) and stops spinning earlier (O2) — ~50% token reduction
4. `erlm_o1o2o3` adds parallel sub-queries (O3) — wall-clock speedup without additional token cost
5. Fine-tuned variants expected to show higher EM across all configurations

**Interpretation note:** EM on 5M-char documents with 5 iterations is inherently low for all methods — the task is extremely hard. The paper's contribution is not "we get higher accuracy" but "we get the same accuracy at half the cost."
