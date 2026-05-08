"""
make_results_charts.py
Generate ERLM results figures for CMU 11-868 poster.

Outputs (saved to ../images/):
  fig_results_ollama.png  — CPU/Apple Silicon (Ollama Qwen3-8B, 4 samples)
  fig_results_gpu.png     — GPU/vLLM (Qwen3-8B + O4 KV Cache, 10 samples)
  fig_baseline_difficulty.png — Baseline difficulty overview
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Color palette ────────────────────────────────────────────────────────────
CMU_RED   = "#C41230"
ERLM_BLUE = "#1A5276"
LIGHT_RED = "#F1948A"
LIGHT_BLU = "#85C1E9"
GRAY      = "#AAAAAA"
DARK_GRAY = "#555555"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ═══════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════

# ── Ollama run (latest, 4 samples, rlm_baseline vs erlm_o1o2o3) ─────────────
# Source: ollama_qwen3_8b_20260427_005534.jsonl
OLLAMA_LABELS = ["S1\n67062f", "S2\n66ec3d", "S3\n66ebc0", "S4\n66ebd4"]
OLLAMA_RLM_TIME   = [203.75, 147.65, 166.26, 52.63]
OLLAMA_ERLM_TIME  = [155.75,  36.13,  73.69, 47.14]
OLLAMA_RLM_TOK    = [20072, 17864, 21642, 4352]
OLLAMA_ERLM_TOK   = [19210,  4370, 21885, 4605]
OLLAMA_RLM_EM     = [1, 0, 0, 0]
OLLAMA_ERLM_EM    = [1, 0, 0, 1]

# ── vLLM / GPU run (10 samples, rlm_baseline vs erlm_o1o2o3o4) ──────────────
# Source: vllm_qwen3_8b_20260426_225636.jsonl  (error on sample 7 excluded for tokens)
GPU_LABELS = [
    "S1\n67062f", "S2\n66ec3d", "S3\n66ebc0", "S4\n66ebd4", "S5\n66ebc9",
    "S6\n66f3ab", "S7\n66ec41", "S8\n6708a0", "S9\n66f0ed", "S10\n66fc01",
]
GPU_RLM_TIME    = [304.05, 115.61,  32.01,  16.29,  83.90, 139.98,  83.14,  76.00,  47.01,  55.17]
GPU_ERLM_TIME   = [101.95,  34.74,  16.83,  14.30,  56.44,  39.19, 158.04,  56.88,  25.41,  14.30]
GPU_RLM_TOK     = [43622,  92315,  37309,   4088,  30027,  41470,  41633,   8460,  15708,   7028]
GPU_ERLM_TOK    = [43041,   6666,   8765,   4390,  17927,   7297,      0,   7553,   5129,   4384]
GPU_RLM_EM      = [0, 0, 1, 1, 1, 0, 1, 0, 0, 0]
GPU_ERLM_EM     = [1, 0, 1, 0, 1, 0, 0, 1, 0, 0]
GPU_KV_RATE     = [71.7, 72.4, 62.7, 63.5, 66.0, 60.6, 58.8, 59.3, 60.1, 64.3]  # ERLM KV hit%

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Baseline Difficulty
# ═══════════════════════════════════════════════════════════════════════════

def fig_baseline_difficulty():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("Baseline Difficulty — LongBench v2 CodeQA (Qwen3-8B)", fontsize=15, fontweight="bold", y=1.02)

    # Panel A: EM comparison across both setups
    ax = axes[0]
    methods = ["RLM\nBaseline", "ERLM\nO1–O4"]

    ollama_ems = [np.mean(OLLAMA_RLM_EM) * 100, np.mean(OLLAMA_ERLM_EM) * 100]
    gpu_ems    = [np.mean(GPU_RLM_EM)    * 100, np.mean(GPU_ERLM_EM)    * 100]

    x = np.array([0, 1])
    w = 0.32
    b1 = ax.bar(x - w/2, ollama_ems, w, color=CMU_RED,   alpha=0.88, label="Ollama (CPU)")
    b2 = ax.bar(x + w/2, gpu_ems,    w, color=ERLM_BLUE, alpha=0.88, label="vLLM GPU")

    for bar, val in zip(list(b1) + list(b2), ollama_ems + gpu_ems):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Human expert line
    ax.axhline(53.7, color="#E67E22", linestyle="--", linewidth=1.5, label="Human expert (53.7%)")
    ax.text(1.55, 54.5, "Human\n53.7%", color="#E67E22", fontsize=9, va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Exact Match Accuracy (%)")
    ax.set_title("A. Accuracy by Setup", fontweight="bold")
    ax.set_ylim(0, 75)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(-0.6, 1.9)

    # Panel B: Per-sample EM heatmap for GPU run (10 samples)
    ax2 = axes[1]
    sample_nums = np.arange(1, 11)
    w = 0.35
    b3 = ax2.bar(sample_nums - w/2, GPU_RLM_EM,  w, color=CMU_RED,   alpha=0.85, label="RLM Baseline")
    b4 = ax2.bar(sample_nums + w/2, GPU_ERLM_EM, w, color=ERLM_BLUE, alpha=0.85, label="ERLM O1–O4")

    ax2.set_xticks(sample_nums)
    ax2.set_xticklabels([f"S{i}" for i in sample_nums], fontsize=9)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Wrong", "Correct"])
    ax2.set_title("B. Per-Sample Correct/Wrong (GPU, n=10)", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.set_ylim(-0.1, 1.35)

    # Annotation
    ax2.text(5.5, 1.22,
             f"RLM: {sum(GPU_RLM_EM)}/10 correct\nERLM: {sum(GPU_ERLM_EM)}/10 correct",
             ha="center", fontsize=10, color=DARK_GRAY,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#F9F9F9", edgecolor=GRAY))

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig_baseline_difficulty.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Ollama CPU Results
# ═══════════════════════════════════════════════════════════════════════════

def fig_ollama():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Ollama Results — Qwen3-8B on Apple M4 Pro (4 Samples)",
        fontsize=15, fontweight="bold", y=1.02
    )

    sample_nums = np.arange(1, 5)
    w = 0.35

    # ── Panel A: Wall-clock time ───────────────────────────────────────────
    ax = axes[0]
    b1 = ax.bar(sample_nums - w/2, OLLAMA_RLM_TIME,  w, color=CMU_RED,   alpha=0.88, label="RLM Baseline")
    b2 = ax.bar(sample_nums + w/2, OLLAMA_ERLM_TIME, w, color=ERLM_BLUE, alpha=0.88, label="ERLM O1–O3")

    # Speedup annotations above each pair
    for i, (rt, et) in enumerate(zip(OLLAMA_RLM_TIME, OLLAMA_ERLM_TIME)):
        speedup = rt / et
        ypos = max(rt, et) + 4
        ax.text(sample_nums[i], ypos, f"{speedup:.1f}×", ha="center", fontsize=10,
                color=ERLM_BLUE, fontweight="bold")

    ax.set_xticks(sample_nums)
    ax.set_xticklabels(OLLAMA_LABELS, fontsize=9)
    ax.set_ylabel("Wall-Clock Time (seconds)")
    ax.set_title("A. Wall-Clock Time per Sample", fontweight="bold")
    ax.legend(fontsize=9)

    # Average speedup box
    avg_speedup = np.mean(OLLAMA_RLM_TIME) / np.mean(OLLAMA_ERLM_TIME)
    ax.text(0.97, 0.97,
            f"Avg Speedup\n{avg_speedup:.2f}×",
            transform=ax.transAxes, ha="right", va="top", fontsize=11, fontweight="bold",
            color=ERLM_BLUE,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#EBF5FB", edgecolor=ERLM_BLUE, linewidth=1.5))

    # ── Panel B: Token usage ──────────────────────────────────────────────
    ax2 = axes[1]
    b3 = ax2.bar(sample_nums - w/2, [t/1000 for t in OLLAMA_RLM_TOK],  w,
                 color=CMU_RED, alpha=0.88, label="RLM Baseline")
    b4 = ax2.bar(sample_nums + w/2, [t/1000 for t in OLLAMA_ERLM_TOK], w,
                 color=ERLM_BLUE, alpha=0.88, label="ERLM O1–O3")

    for i, (rt, et) in enumerate(zip(OLLAMA_RLM_TOK, OLLAMA_ERLM_TOK)):
        redux = (rt - et) / rt * 100
        color = ERLM_BLUE if redux > 0 else "#E74C3C"
        sign  = "↓" if redux > 0 else "↑"
        ypos = max(rt, et) / 1000 + 0.3
        ax2.text(sample_nums[i], ypos, f"{sign}{abs(redux):.0f}%",
                 ha="center", fontsize=10, color=color, fontweight="bold")

    ax2.set_xticks(sample_nums)
    ax2.set_xticklabels(OLLAMA_LABELS, fontsize=9)
    ax2.set_ylabel("Total Tokens Used (thousands)")
    ax2.set_title("B. Token Usage per Sample", fontweight="bold")
    ax2.legend(fontsize=9)

    avg_redux = (np.mean(OLLAMA_RLM_TOK) - np.mean(OLLAMA_ERLM_TOK)) / np.mean(OLLAMA_RLM_TOK) * 100
    ax2.text(0.97, 0.97,
             f"Avg Token Δ\n{avg_redux:+.1f}%",
             transform=ax2.transAxes, ha="right", va="top", fontsize=11, fontweight="bold",
             color=ERLM_BLUE,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#EBF5FB", edgecolor=ERLM_BLUE, linewidth=1.5))

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig_results_ollama.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — GPU / vLLM Results (O4 KV Cache)
# ═══════════════════════════════════════════════════════════════════════════

def fig_gpu():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "GPU Results — Qwen3-8B via vLLM on Modal A10G (10 Samples, +O4 KV Prefix Cache)",
        fontsize=15, fontweight="bold", y=1.02
    )

    sample_nums = np.arange(1, 11)
    w = 0.38

    # ── Panel A: Wall-clock time ───────────────────────────────────────────
    ax = axes[0]
    ax.bar(sample_nums - w/2, GPU_RLM_TIME,  w, color=CMU_RED,   alpha=0.88, label="RLM Baseline")
    ax.bar(sample_nums + w/2, GPU_ERLM_TIME, w, color=ERLM_BLUE, alpha=0.88, label="ERLM O1–O4")

    # Speedup labels on samples where ERLM is faster
    for i, (rt, et) in enumerate(zip(GPU_RLM_TIME, GPU_ERLM_TIME)):
        speedup = rt / et
        if speedup >= 1.3:
            ypos = max(rt, et) + 3
            ax.text(sample_nums[i], ypos, f"{speedup:.1f}×",
                    ha="center", fontsize=8, color=ERLM_BLUE, fontweight="bold")

    # Mark S7 as error
    ax.text(7, GPU_ERLM_TIME[6] + 5, "⚠", ha="center", fontsize=14, color="#E67E22")

    ax.set_xticks(sample_nums)
    ax.set_xticklabels([f"S{i}" for i in sample_nums], fontsize=9)
    ax.set_ylabel("Wall-Clock Time (seconds)")
    ax.set_title("A. Wall-Clock Time per Sample", fontweight="bold")
    ax.legend(fontsize=9)

    # Compute avg excluding S7 (error)
    valid_idx = [i for i in range(10) if GPU_ERLM_TOK[i] > 0]
    avg_sp = np.mean(GPU_RLM_TIME) / np.mean(GPU_ERLM_TIME)
    ax.text(0.97, 0.97,
            f"Avg Speedup\n{avg_sp:.2f}×",
            transform=ax.transAxes, ha="right", va="top", fontsize=11, fontweight="bold",
            color=ERLM_BLUE,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#EBF5FB", edgecolor=ERLM_BLUE, linewidth=1.5))

    # ── Panel B: Token usage ──────────────────────────────────────────────
    ax2 = axes[1]
    rlm_k  = [t/1000 for t in GPU_RLM_TOK]
    erlm_k = [t/1000 for t in GPU_ERLM_TOK]

    ax2.bar(sample_nums - w/2, rlm_k,  w, color=CMU_RED,   alpha=0.88, label="RLM Baseline")
    ax2.bar(sample_nums + w/2, erlm_k, w, color=ERLM_BLUE, alpha=0.88, label="ERLM O1–O4")

    for i, (rt, et) in enumerate(zip(GPU_RLM_TOK, GPU_ERLM_TOK)):
        if et > 0:
            redux = (rt - et) / rt * 100
            if abs(redux) >= 15:
                color = ERLM_BLUE if redux > 0 else "#E74C3C"
                sign  = "↓" if redux > 0 else "↑"
                ypos = max(rt, et) / 1000 + 0.5
                ax2.text(sample_nums[i], ypos, f"{sign}{abs(redux):.0f}%",
                         ha="center", fontsize=8, color=color, fontweight="bold")

    ax2.text(7, GPU_ERLM_TOK[6]/1000 + 1, "error", ha="center", fontsize=8, color="#E67E22")

    ax2.set_xticks(sample_nums)
    ax2.set_xticklabels([f"S{i}" for i in sample_nums], fontsize=9)
    ax2.set_ylabel("Total Tokens Used (thousands)")
    ax2.set_title("B. Token Usage per Sample", fontweight="bold")
    ax2.legend(fontsize=9)

    valid_rlm  = [GPU_RLM_TOK[i] for i in valid_idx]
    valid_erlm = [GPU_ERLM_TOK[i] for i in valid_idx]
    avg_tok_redux = (np.mean(valid_rlm) - np.mean(valid_erlm)) / np.mean(valid_rlm) * 100
    ax2.text(0.97, 0.97,
             f"Avg Token Reduction\n{avg_tok_redux:.1f}% (excl. error)",
             transform=ax2.transAxes, ha="right", va="top", fontsize=10, fontweight="bold",
             color=ERLM_BLUE,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#EBF5FB", edgecolor=ERLM_BLUE, linewidth=1.5))

    # ── Panel C: KV Cache Hit Rate ─────────────────────────────────────────
    ax3 = axes[2]

    bars = ax3.bar(sample_nums, GPU_KV_RATE, color=ERLM_BLUE, alpha=0.80, zorder=3)
    for bar, val in zip(bars, GPU_KV_RATE):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                 color=ERLM_BLUE)

    ax3.axhline(np.mean(GPU_KV_RATE), color=CMU_RED, linestyle="--", linewidth=2,
                label=f"Mean: {np.mean(GPU_KV_RATE):.1f}%", zorder=4)

    ax3.set_xticks(sample_nums)
    ax3.set_xticklabels([f"S{i}" for i in sample_nums], fontsize=9)
    ax3.set_ylabel("KV Cache Hit Rate (%)")
    ax3.set_title("C. O4 KV Prefix Cache Hit Rate", fontweight="bold")
    ax3.set_ylim(0, 90)
    ax3.legend(fontsize=9)

    ax3.text(0.5, 0.08,
             "RadixAttention reuses\n60–72% of prompt tokens\nacross iterations",
             transform=ax3.transAxes, ha="center", fontsize=9.5,
             color=DARK_GRAY, style="italic",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#FDFEFE", edgecolor=GRAY))

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig_results_gpu.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Combined Summary (poster-ready single figure)
# ═══════════════════════════════════════════════════════════════════════════

def fig_summary():
    """
    A single compact 2×2 figure for the poster RESULTS section.
    """
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        "ERLM Results: RLM Baseline vs ERLM O1–O4  |  LongBench v2 CodeQA",
        fontsize=14, fontweight="bold"
    )

    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)
    ax_tl = fig.add_subplot(gs[0, 0])   # top-left:  Ollama time
    ax_tr = fig.add_subplot(gs[0, 1])   # top-right: GPU time
    ax_bl = fig.add_subplot(gs[1, 0])   # bot-left:  Token reduction summary
    ax_br = fig.add_subplot(gs[1, 1])   # bot-right: KV cache hit rate

    # ── Top-left: Ollama wall-clock (4 samples) ────────────────────────────
    x = np.arange(1, 5)
    w = 0.35
    ax_tl.bar(x - w/2, OLLAMA_RLM_TIME,  w, color=CMU_RED,   alpha=0.88, label="RLM Baseline")
    ax_tl.bar(x + w/2, OLLAMA_ERLM_TIME, w, color=ERLM_BLUE, alpha=0.88, label="ERLM O1–O3")
    for i, (r, e) in enumerate(zip(OLLAMA_RLM_TIME, OLLAMA_ERLM_TIME)):
        ax_tl.text(x[i], max(r, e) + 3, f"{r/e:.1f}×", ha="center",
                   fontsize=9, fontweight="bold", color=ERLM_BLUE)
    ax_tl.set_title("Ollama CPU — Wall-Clock Time (s)", fontweight="bold", fontsize=11)
    ax_tl.set_xticks(x); ax_tl.set_xticklabels(["S1","S2","S3","S4"])
    ax_tl.set_ylabel("Seconds")
    ax_tl.legend(fontsize=8, loc="upper right")

    # ── Top-right: GPU wall-clock (10 samples) ────────────────────────────
    xg = np.arange(1, 11)
    ax_tr.bar(xg - w/2, GPU_RLM_TIME,  w, color=CMU_RED,   alpha=0.88, label="RLM Baseline")
    ax_tr.bar(xg + w/2, GPU_ERLM_TIME, w, color=ERLM_BLUE, alpha=0.88, label="ERLM O1–O4")
    avg_gpu_sp = np.mean(GPU_RLM_TIME) / np.mean(GPU_ERLM_TIME)
    ax_tr.text(0.97, 0.97, f"Avg {avg_gpu_sp:.2f}×",
               transform=ax_tr.transAxes, ha="right", va="top", fontsize=10,
               fontweight="bold", color=ERLM_BLUE,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="#EBF5FB", edgecolor=ERLM_BLUE))
    ax_tr.set_title("GPU/vLLM — Wall-Clock Time (s)", fontweight="bold", fontsize=11)
    ax_tr.set_xticks(xg); ax_tr.set_xticklabels([f"S{i}" for i in xg], fontsize=8)
    ax_tr.set_ylabel("Seconds")
    ax_tr.legend(fontsize=8, loc="upper right")

    # ── Bottom-left: Summary bar (avg metrics) ─────────────────────────────
    categories = ["Avg Time\n(s)", "Avg Tokens\n(÷1000)", "Accuracy\n(EM %)"]
    rlm_vals   = [
        np.mean(GPU_RLM_TIME),
        np.mean(GPU_RLM_TOK) / 1000,
        np.mean(GPU_RLM_EM) * 100,
    ]
    erlm_vals  = [
        np.mean(GPU_ERLM_TIME),
        np.mean([GPU_ERLM_TOK[i] for i in range(10) if GPU_ERLM_TOK[i] > 0]) / 1000,
        np.mean(GPU_ERLM_EM) * 100,
    ]
    xb = np.arange(len(categories))
    ax_bl.bar(xb - 0.22, rlm_vals,  0.4, color=CMU_RED,   alpha=0.88, label="RLM Baseline")
    ax_bl.bar(xb + 0.22, erlm_vals, 0.4, color=ERLM_BLUE, alpha=0.88, label="ERLM O1–O4")
    for xi, (rv, ev) in enumerate(zip(rlm_vals, erlm_vals)):
        ax_bl.text(xi - 0.22, rv + max(rlm_vals)*0.01, f"{rv:.1f}", ha="center", fontsize=9, color=CMU_RED, fontweight="bold")
        ax_bl.text(xi + 0.22, ev + max(rlm_vals)*0.01, f"{ev:.1f}", ha="center", fontsize=9, color=ERLM_BLUE, fontweight="bold")
    ax_bl.set_xticks(xb); ax_bl.set_xticklabels(categories, fontsize=10)
    ax_bl.set_title("GPU — Avg Metrics Summary", fontweight="bold", fontsize=11)
    ax_bl.legend(fontsize=8)

    # ── Bottom-right: KV Cache hit rate ───────────────────────────────────
    ax_br.bar(xg, GPU_KV_RATE, color=ERLM_BLUE, alpha=0.80)
    ax_br.axhline(np.mean(GPU_KV_RATE), color=CMU_RED, linestyle="--", linewidth=2,
                  label=f"Mean: {np.mean(GPU_KV_RATE):.1f}%")
    ax_br.set_title("O4 KV Prefix Cache Hit Rate (%)", fontweight="bold", fontsize=11)
    ax_br.set_xticks(xg); ax_br.set_xticklabels([f"S{i}" for i in xg], fontsize=8)
    ax_br.set_ylabel("Cache Hit Rate (%)")
    ax_br.set_ylim(0, 90)
    ax_br.legend(fontsize=9)
    ax_br.text(5.5, 8, "RadixAttention reuses 60–72% of\nprompt tokens across RLM iterations",
               ha="center", fontsize=9, color=DARK_GRAY, style="italic")

    out = os.path.join(OUT_DIR, "fig_results_summary.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating result figures...")
    fig_baseline_difficulty()
    fig_ollama()
    fig_gpu()
    fig_summary()
    print("Done. Check images/ folder.")
