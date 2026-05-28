from transformers import AutoProcessor
import json
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random

dataset_path = Path("ShaderDataset/train")
img_list = list(dataset_path.rglob("*.jpg"))

# sample only 20k files instead of all 350k
SAMPLE_SIZE = 20000
if len(img_list) > SAMPLE_SIZE:
    img_list = random.sample(img_list, SAMPLE_SIZE)
    print(f"Sampling {SAMPLE_SIZE} / {len(img_list) + SAMPLE_SIZE} files for speed")

processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B")

old_seq_lengths = []
new_seq_lengths = []

for img in tqdm(img_list):
    txt_path = img.with_suffix(".txt")
    python_path = Path("material_dataset_filtered").joinpath(*img.parts[2:-1]) / f"var_{img.stem}_full.py"

    with open(python_path, "r") as f:
        code = f.read()

    old_seq_lengths.append(len(processor.tokenizer.tokenize(code)))

    with open(txt_path, "r") as f:
        txt = f.read()

    new_seq_lengths.append(len(processor.tokenizer.tokenize(txt)))




# dataset_dir = Path("ShaderDataset")
# img_files = list(dataset_dir.rglob("*.jpg"))


# # -- Getting old sequence lengths ----------------------------------- #
# old_seq_lengths = []
# for img in tqdm(img_files, desc="Old tokenizer"):
#     txt_path = img.with_suffix(".txt")
#     with open(txt_path, "r") as f:
#         shader_text = f.read()
#     old_seq_lengths.append(len(processor.tokenizer.tokenize(shader_text)))

# # -- Adding new tokens -------------------------------------------- #
# new_tokens_json = "JSON_files/addition_tokens.json"
# with open(new_tokens_json, "r") as f:
#     token_dict = json.load(f)

# print(f"old vocab size - {len(processor.tokenizer)}")
# processor.tokenizer.add_tokens(token_dict["new_tokens"])
# processor.tokenizer.add_special_tokens({
#     "additional_special_tokens": token_dict["special_tokens"]
# })
# print(f"new vocab size - {len(processor.tokenizer)}")

# # -- Getting new sequence lengths ----------------------------------- #
# new_seq_lengths = []
# for img in tqdm(img_files, desc="New tokenizer"):
#     txt_path = img.with_suffix(".txt")
#     with open(txt_path, "r") as f:
#         shader_text = f.read()
#     new_seq_lengths.append(len(processor.tokenizer.tokenize(shader_text)))

# -- Stats --------------------------------------------------------- #
def stats(arr):
    arr = np.array(arr)
    return {
        "min": int(arr.min()), "max": int(arr.max()),
        "mean": arr.mean(), "median": np.median(arr),
        "p90": np.percentile(arr, 90), "p95": np.percentile(arr, 95),
        "under_512":  (arr <= 512).mean()  * 100,
        "under_1024": (arr <= 1024).mean() * 100,
    }

os = stats(old_seq_lengths)
ns = stats(new_seq_lengths)
reduction = np.array(old_seq_lengths) - np.array(new_seq_lengths)

print(f"\n{'Metric':<15} {'Old':>10} {'New':>10}")
print("-" * 37)
for k in ["min", "max", "mean", "median", "p90", "p95"]:
    print(f"{k:<15} {os[k]:>10.1f} {ns[k]:>10.1f}")
print(f"{'<=512 %':<15} {os['under_512']:>9.1f}% {ns['under_512']:>9.1f}%")
print(f"{'<=1024 %':<15} {os['under_1024']:>9.1f}% {ns['under_1024']:>9.1f}%")
print(f"\nAvg token reduction: {reduction.mean():.1f} tokens/sample")

# -- Visualization -------------------------------------- #
fig, axes = plt.subplots(1, 3, figsize=(16, 4), facecolor="#111")
fig.suptitle("Token Sequence Length Analysis between python code & text shader", color="#eee", fontsize=14)

bins = np.linspace(0, max(max(old_seq_lengths), max(new_seq_lengths)), 60)

# histograms
ax = axes[0]
ax.hist(old_seq_lengths, bins=bins, color="#4fc3f7", alpha=0.6, label="Old")
ax.hist(new_seq_lengths, bins=bins, color="#f06292", alpha=0.6, label="New")
ax.axvline(np.mean(old_seq_lengths),  color="#ef5350", linestyle="--", linewidth=1.2, label=f"avg {np.mean(old_seq_lengths):.1f}")
ax.axvline(np.mean(old_seq_lengths), color="#ab47bc", linestyle="--", linewidth=1.2, label=f"avg {np.mean(new_seq_lengths):.1f}")
ax.set_title("Distribution", color="#eee")
ax.legend(fontsize=8, facecolor="#222", labelcolor="#ddd")
ax.set_facecolor("#1a1a1a")
ax.tick_params(colors="#888")

# CDF
ax = axes[1]
for lengths, color, label in [(old_seq_lengths, "#4fc3f7", "Old"),
                                (new_seq_lengths, "#f06292", "New")]:
    s = np.sort(lengths)
    cdf = np.arange(1, len(s)+1) / len(s) * 100
    ax.plot(s, cdf, color=color, linewidth=2, label=label)
ax.axvline(np.mean(old_seq_lengths),  color="#ef5350", linestyle="--", linewidth=1.2, label=f"avg {np.mean(old_seq_lengths):.1f}")
ax.axvline(np.mean(new_seq_lengths), color="#ab47bc", linestyle="--", linewidth=1.2, label=f"avg {np.mean(new_seq_lengths):.1f}")
ax.axhline(90, color="#555", linestyle=":", linewidth=1)
ax.axhline(95, color="#555", linestyle=":", linewidth=1)
ax.set_title("Cumulative %", color="#eee")
ax.set_ylabel("% samples", color="#888")
ax.legend(fontsize=8, facecolor="#222", labelcolor="#ddd")
ax.set_facecolor("#1a1a1a")
ax.tick_params(colors="#888")

# reduction histogram
ax = axes[2]
ax.hist(reduction, bins=50, color="#81c784", alpha=0.85, edgecolor="none")
# ax.axvline(0, color="#ef5350", linewidth=1.2)
ax.axvline(reduction.mean(), color="#ffeb3b", linestyle="--",
           linewidth=1.5, label=f"Avg: {reduction.mean():.1f}")
ax.set_title("Token Reduction", color="#eee")
ax.set_xlabel("tokens saved per sample", color="#888")
ax.legend(fontsize=8, facecolor="#222", labelcolor="#ddd")
ax.set_facecolor("#1a1a1a")
ax.tick_params(colors="#888")

for ax in axes:
    ax.grid(True, color="#2a2a2a", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

plt.tight_layout()
plt.savefig("python_to_text_token_analysis.png", dpi=120, bbox_inches="tight", facecolor="#111")
print("\nSaved to python_to_text_token_analysis.png")
plt.show()

""" 

Metric                 Old        New
-------------------------------------
min                  158.0       65.0
max                 2470.0     1747.0
mean                 907.2      577.9
median               886.0      562.0
p90                 1189.1      784.0
p95                 1300.0      866.0
<=512 %               2.8%      35.8%
<=1024 %             74.3%      98.8%

Avg token reduction: 329.3 tokens/sample
Saved to python_to_text_token_analysis.png


old vocab size - 248077
new vocab size - 248310

Metric                 Old        New
-------------------------------------
min                   65.0       49.0
max                 1833.0     1597.0
mean                 583.8      473.7
median               557.0      450.0
p90                  796.1      657.0
p95                  904.0      751.0
<=512 %              37.1%      67.4%
<=1024 %             97.7%      98.7%

Avg token reduction: 110.1 tokens/sample
"""