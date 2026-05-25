from transformers import AutoProcessor
import json
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

dataset_dir = Path("ShaderDataset")
img_files = list(dataset_dir.rglob("*.jpg"))

processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B")

# -- Getting old sequence lengths ----------------------------------- #
old_seq_lengths = []
for img in tqdm(img_files, desc="Old tokenizer"):
    txt_path = img.with_suffix(".txt")
    with open(txt_path, "r") as f:
        shader_text = f.read()
    tokens = processor.tokenizer.tokenize(shader_text)
    old_seq_lengths.append(len(tokens))

# -- Adding new tokens to tokenizer ------------------------------------ #
new_tokens_json = "JSON_files/addition_tokens.json"
with open(new_tokens_json, "r") as f:
    token_dict = json.load(f)

print(f"old token vocab size - {len(processor.tokenizer)}")
processor.tokenizer.add_tokens(token_dict["new_tokens"])
processor.tokenizer.add_special_tokens({
    "additional_special_tokens": token_dict["special_tokens"]
})
print(f"new token vocab size - {len(processor.tokenizer)}")

# -- Getting new sequence lengths ----------------------------------- #
seq_lengths = []
for img in tqdm(img_files, desc="New tokenizer"):
    txt_path = img.with_suffix(".txt")
    with open(txt_path, "r") as f:
        shader_text = f.read()
    tokens = processor.tokenizer.tokenize(shader_text)
    seq_lengths.append(len(tokens))

# -- Stats computation ---------------------------------------------- #
def get_stats(lengths):
    arr = np.array(lengths)
    return {
        "min":    int(arr.min()),
        "max":    int(arr.max()),
        "mean":   float(arr.mean()),
        "median": float(np.median(arr)),
        "std":    float(arr.std()),
        "p25":    float(np.percentile(arr, 25)),
        "p75":    float(np.percentile(arr, 75)),
        "p90":    float(np.percentile(arr, 90)),
        "p95":    float(np.percentile(arr, 95)),
        "p99":    float(np.percentile(arr, 99)),
        "under_512":  int((arr <= 512).sum()),
        "under_1024": int((arr <= 1024).sum()),
        "over_1024":  int((arr > 1024).sum()),
        "total":  len(arr),
    }

old_stats = get_stats(old_seq_lengths)
new_stats = get_stats(seq_lengths)

reduction = np.array(old_seq_lengths) - np.array(seq_lengths)

# -- Visualization ---------------------------------------------- #
plt.style.use("dark_background")

fig = plt.figure(figsize=(20, 24), facecolor="#0d0d0d")
fig.suptitle("Token Sequence Length Analysis\nOld vs New Tokenizer", 
             fontsize=22, fontweight="bold", color="#e0e0e0", y=0.98)

gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

OLD_COLOR  = "#4fc3f7"
NEW_COLOR  = "#f06292"
DIFF_COLOR = "#81c784"
GRID_COLOR = "#2a2a2a"

# ── 1. Distribution histograms side by side ───────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

bins = np.linspace(
    min(min(old_seq_lengths), min(seq_lengths)),
    max(max(old_seq_lengths), max(seq_lengths)),
    60
)

ax1.hist(old_seq_lengths, bins=bins, color=OLD_COLOR, alpha=0.85, edgecolor="#0d0d0d", linewidth=0.4)
ax1.axvline(old_stats["mean"],   color="#ffeb3b", linestyle="--", linewidth=1.5, label=f'Mean {old_stats["mean"]:.0f}')
ax1.axvline(old_stats["median"], color="#ff9800", linestyle=":",  linewidth=1.5, label=f'Median {old_stats["median"]:.0f}')
ax1.axvline(512,  color="#ef5350", linestyle="-", linewidth=1.2, alpha=0.7, label="512 cutoff")
ax1.axvline(1024, color="#ab47bc", linestyle="-", linewidth=1.2, alpha=0.7, label="1024 cutoff")
ax1.set_title("Old Tokenizer Distribution", color=OLD_COLOR, fontsize=13, pad=8)
ax1.set_xlabel("Sequence Length", color="#aaa")
ax1.set_ylabel("Count", color="#aaa")
ax1.legend(fontsize=8, facecolor="#1a1a1a", edgecolor="#333", labelcolor="#ddd")
ax1.grid(True, color=GRID_COLOR, linewidth=0.5)
ax1.tick_params(colors="#888")
for spine in ax1.spines.values(): spine.set_edgecolor("#333")

ax2.hist(seq_lengths, bins=bins, color=NEW_COLOR, alpha=0.85, edgecolor="#0d0d0d", linewidth=0.4)
ax2.axvline(new_stats["mean"],   color="#ffeb3b", linestyle="--", linewidth=1.5, label=f'Mean {new_stats["mean"]:.0f}')
ax2.axvline(new_stats["median"], color="#ff9800", linestyle=":",  linewidth=1.5, label=f'Median {new_stats["median"]:.0f}')
ax2.axvline(512,  color="#ef5350", linestyle="-", linewidth=1.2, alpha=0.7, label="512 cutoff")
ax2.axvline(1024, color="#ab47bc", linestyle="-", linewidth=1.2, alpha=0.7, label="1024 cutoff")
ax2.set_title("New Tokenizer Distribution", color=NEW_COLOR, fontsize=13, pad=8)
ax2.set_xlabel("Sequence Length", color="#aaa")
ax2.set_ylabel("Count", color="#aaa")
ax2.legend(fontsize=8, facecolor="#1a1a1a", edgecolor="#333", labelcolor="#ddd")
ax2.grid(True, color=GRID_COLOR, linewidth=0.5)
ax2.tick_params(colors="#888")
for spine in ax2.spines.values(): spine.set_edgecolor("#333")

# ── 2. Overlapping KDE-style comparison ──────────────────────────
ax3 = fig.add_subplot(gs[1, :])

ax3.hist(old_seq_lengths, bins=80, color=OLD_COLOR, alpha=0.5, label="Old tokenizer", edgecolor="none")
ax3.hist(seq_lengths,     bins=80, color=NEW_COLOR, alpha=0.5, label="New tokenizer", edgecolor="none")
ax3.axvline(512,  color="#ef5350", linestyle="--", linewidth=1.5, alpha=0.8, label="512")
ax3.axvline(1024, color="#ab47bc", linestyle="--", linewidth=1.5, alpha=0.8, label="1024")
ax3.set_title("Overlapping Distribution Comparison", color="#e0e0e0", fontsize=13, pad=8)
ax3.set_xlabel("Sequence Length", color="#aaa")
ax3.set_ylabel("Count", color="#aaa")
ax3.legend(fontsize=10, facecolor="#1a1a1a", edgecolor="#333", labelcolor="#ddd")
ax3.grid(True, color=GRID_COLOR, linewidth=0.5)
ax3.tick_params(colors="#888")
for spine in ax3.spines.values(): spine.set_edgecolor("#333")

# ── 3. Token reduction per sample ────────────────────────────────
ax4 = fig.add_subplot(gs[2, :])

sorted_idx = np.argsort(reduction)[::-1]
colors_bar = [DIFF_COLOR if r >= 0 else "#ef5350" for r in np.array(reduction)[sorted_idx]]
ax4.bar(range(len(reduction)), np.array(reduction)[sorted_idx], color=colors_bar, alpha=0.8, width=1.0)
ax4.axhline(0, color="#888", linewidth=0.8)
ax4.axhline(reduction.mean(), color="#ffeb3b", linestyle="--", linewidth=1.5, label=f"Avg reduction: {reduction.mean():.1f}")
ax4.set_title("Token Reduction per Sample (Old − New)", color="#e0e0e0", fontsize=13, pad=8)
ax4.set_xlabel("Sample index (sorted by reduction)", color="#aaa")
ax4.set_ylabel("Tokens saved", color="#aaa")
ax4.legend(fontsize=10, facecolor="#1a1a1a", edgecolor="#333", labelcolor="#ddd")
ax4.grid(True, color=GRID_COLOR, linewidth=0.5, axis="y")
ax4.tick_params(colors="#888")
for spine in ax4.spines.values(): spine.set_edgecolor("#333")

# ── 4. Stats table ────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[3, 0])
ax5.axis("off")

labels = ["Min", "Max", "Mean", "Median", "Std Dev",
          "P25", "P75", "P90", "P95", "P99",
          "≤512", "≤1024", ">1024", "Total"]
old_vals = [
    old_stats["min"], old_stats["max"], f'{old_stats["mean"]:.1f}',
    f'{old_stats["median"]:.1f}', f'{old_stats["std"]:.1f}',
    f'{old_stats["p25"]:.0f}', f'{old_stats["p75"]:.0f}',
    f'{old_stats["p90"]:.0f}', f'{old_stats["p95"]:.0f}', f'{old_stats["p99"]:.0f}',
    f'{old_stats["under_512"]} ({100*old_stats["under_512"]/old_stats["total"]:.1f}%)',
    f'{old_stats["under_1024"]} ({100*old_stats["under_1024"]/old_stats["total"]:.1f}%)',
    f'{old_stats["over_1024"]} ({100*old_stats["over_1024"]/old_stats["total"]:.1f}%)',
    old_stats["total"]
]
new_vals = [
    new_stats["min"], new_stats["max"], f'{new_stats["mean"]:.1f}',
    f'{new_stats["median"]:.1f}', f'{new_stats["std"]:.1f}',
    f'{new_stats["p25"]:.0f}', f'{new_stats["p75"]:.0f}',
    f'{new_stats["p90"]:.0f}', f'{new_stats["p95"]:.0f}', f'{new_stats["p99"]:.0f}',
    f'{new_stats["under_512"]} ({100*new_stats["under_512"]/new_stats["total"]:.1f}%)',
    f'{new_stats["under_1024"]} ({100*new_stats["under_1024"]/new_stats["total"]:.1f}%)',
    f'{new_stats["over_1024"]} ({100*new_stats["over_1024"]/new_stats["total"]:.1f}%)',
    new_stats["total"]
]

table = ax5.table(
    cellText=[[l, str(o), str(n)] for l, o, n in zip(labels, old_vals, new_vals)],
    colLabels=["Metric", "Old Tokenizer", "New Tokenizer"],
    loc="center",
    cellLoc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.6)

for (row, col), cell in table.get_celld().items():
    cell.set_facecolor("#1a1a1a" if row == 0 else ("#0d0d0d" if row % 2 == 0 else "#141414"))
    cell.set_text_props(color=OLD_COLOR if col == 1 and row > 0 else
                              NEW_COLOR if col == 2 and row > 0 else "#e0e0e0")
    cell.set_edgecolor("#2a2a2a")

ax5.set_title("Statistics Summary", color="#e0e0e0", fontsize=13, pad=12)

# ── 5. Cumulative distribution ────────────────────────────────────
ax6 = fig.add_subplot(gs[3, 1])

old_sorted = np.sort(old_seq_lengths)
new_sorted = np.sort(seq_lengths)
old_cdf = np.arange(1, len(old_sorted) + 1) / len(old_sorted)
new_cdf = np.arange(1, len(new_sorted) + 1) / len(new_sorted)

ax6.plot(old_sorted, old_cdf * 100, color=OLD_COLOR, linewidth=2, label="Old tokenizer")
ax6.plot(new_sorted, new_cdf * 100, color=NEW_COLOR, linewidth=2, label="New tokenizer")
ax6.axvline(512,  color="#ef5350", linestyle="--", linewidth=1.2, alpha=0.8, label="512")
ax6.axvline(1024, color="#ab47bc", linestyle="--", linewidth=1.2, alpha=0.8, label="1024")
ax6.axhline(90, color="#888", linestyle=":", linewidth=1, alpha=0.6)
ax6.axhline(95, color="#888", linestyle=":", linewidth=1, alpha=0.6)
ax6.set_title("Cumulative Distribution (%)", color="#e0e0e0", fontsize=13, pad=8)
ax6.set_xlabel("Sequence Length", color="#aaa")
ax6.set_ylabel("% of samples", color="#aaa")
ax6.legend(fontsize=9, facecolor="#1a1a1a", edgecolor="#333", labelcolor="#ddd")
ax6.grid(True, color=GRID_COLOR, linewidth=0.5)
ax6.tick_params(colors="#888")
for spine in ax6.spines.values(): spine.set_edgecolor("#333")

plt.savefig("token_analysis.png", dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
print("Saved to token_analysis.png")

# -- Print summary to console --------------------------------------- #
print("\n" + "="*50)
print(f"{'Metric':<20} {'Old':>12} {'New':>12}")
print("="*50)
for label, o, n in zip(labels, old_vals, new_vals):
    print(f"{label:<20} {str(o):>12} {str(n):>12}")
print("="*50)
print(f"{'Avg token reduction':<20} {reduction.mean():>12.1f}")
print(f"{'Total tokens saved':<20} {int(reduction.sum()):>12,}")