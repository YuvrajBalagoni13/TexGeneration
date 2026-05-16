from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm.auto import tqdm
import json

split_type = "train_2"
data_path = Path(f"ShaderDataset/{split_type}")
dataset_info_json_path = f"ShaderDataset/data_{split_type}_info.json"
plots_output_dir = Path(f"ShaderDataset/{split_type}_distribution_plots")
# data_path = Path("ShaderDataset")
# dataset_info_json_path = f"ShaderDataset/dataset_info.json"
# plots_output_dir = Path(f"ShaderDataset/Dataset_distribution_plots")
plots_output_dir.mkdir(exist_ok=True)

image_list = list(data_path.rglob("*.jpg"))

data_info = {}

data_info["avg_no_of_nodes"] = 0
data_info["avg_no_of_links"] = 0
data_info["nodes_info"] = defaultdict(int)  # FIX: KeyError on +=
data_info["no_of_nodes_outliers"] = 0
data_info["nodes_outliers"] = {}
data_info["links_info"] = defaultdict(int)  # FIX: KeyError on +=
data_info["nodes_distribution"] = []
data_info["links_distribution"] = []

for image in tqdm(image_list):
    txt_file = image.with_suffix(".txt")

    with open(txt_file, "r") as f:
        shader = f.read()

    lines = shader.strip().split("\n")
    for line in lines:
        if line.startswith("N|"):
            node_info = [n for n in line[2:].split(";") if n]
            data_info["nodes_distribution"].append(len(node_info))
            data_info["avg_no_of_nodes"] += len(node_info)
            if len(node_info) > 50:
                data_info["nodes_outliers"][str(image)] = len(node_info)
                data_info["no_of_nodes_outliers"] += 1
            for node in node_info:
                parts = node.split(":")
                node_type = parts[-1].strip() if parts else "unknown" 
                data_info["nodes_info"][node_type] += 1

        if line.startswith("L|"):
            link_info = [n for n in line[2:].split(";") if n]
            data_info["links_distribution"].append(len(link_info))
            data_info["avg_no_of_links"] += len(link_info)
            for link in link_info:
                data_info["links_info"][link.strip()] += 1

# Averages
total = len(image_list)
data_info["avg_no_of_nodes"] = data_info["avg_no_of_nodes"] / total if total else 0
data_info["avg_no_of_links"] = data_info["avg_no_of_links"] / total if total else 0

# Convert defaultdicts to regular dicts for JSON serialization
data_info["nodes_info"] = dict(data_info["nodes_info"])
data_info["links_info"] = dict(data_info["links_info"])

with open(dataset_info_json_path, "w") as f:  # FIX: missing open()
    json.dump(data_info, f, indent=4)

print(f"Avg nodes per sample: {data_info['avg_no_of_nodes']:.2f}")
print(f"Avg links per sample: {data_info['avg_no_of_links']:.2f}")


# ---------- Visualization ---------- #

def save_fig(fig, name):
    path = plots_output_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# 1. Node Count Distribution (histogram)
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(data_info["nodes_distribution"], bins=50, color="steelblue", edgecolor="white")
ax.axvline(data_info["avg_no_of_nodes"], color="red", linestyle="--",
           label=f"Mean: {data_info['avg_no_of_nodes']:.1f}")
ax.set_title(f"{split_type} Node Count Distribution per Sample")
ax.set_xlabel("Number of Nodes")
ax.set_ylabel("Frequency")
ax.legend()
save_fig(fig, f"{split_type}_node_count_distribution.png")


# 2. Link Count Distribution (histogram)
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(data_info["links_distribution"], bins=50, color="darkorange", edgecolor="white")
ax.axvline(data_info["avg_no_of_links"], color="red", linestyle="--",
           label=f"Mean: {data_info['avg_no_of_links']:.1f}")
ax.set_title(f"{split_type} Link Count Distribution per Sample")
ax.set_xlabel("Number of Links")
ax.set_ylabel("Frequency")
ax.legend()
save_fig(fig, f"{split_type}_link_count_distribution.png")


# 3. Top N Node Types (bar chart)
TOP_N = 30
nodes_info = data_info["nodes_info"]
sorted_nodes = sorted(nodes_info.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
node_labels, node_counts = zip(*sorted_nodes) if sorted_nodes else ([], [])

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(range(len(node_labels)), node_counts, color="steelblue")
ax.set_xticks(range(len(node_labels)))
ax.set_xticklabels(node_labels, rotation=45, ha="right", fontsize=8)
ax.set_title(f"{split_type} Top {TOP_N} Most Frequent Node Types")
ax.set_xlabel("Node Type")
ax.set_ylabel("Count")
save_fig(fig, f"{split_type}_top_node_types.png")


# 4. Top N Link Types (bar chart)
links_info = data_info["links_info"]
sorted_links = sorted(links_info.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
link_labels, link_counts = zip(*sorted_links) if sorted_links else ([], [])

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(range(len(link_labels)), link_counts, color="darkorange")
ax.set_xticks(range(len(link_labels)))
ax.set_xticklabels(link_labels, rotation=45, ha="right", fontsize=8)
ax.set_title(f"{split_type} Top {TOP_N} Most Frequent Link Types")
ax.set_xlabel("Link Type")
ax.set_ylabel("Count")
save_fig(fig, f"{split_type}_top_link_types.png")


# 5. Nodes vs Links scatter (complexity overview)
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data_info["nodes_distribution"],
           data_info["links_distribution"],
           alpha=0.2, s=5, color="purple")

# ----------------------------------------------- #
# ax.set_xscale("log")
# ax.set_yscale("log")
# ----------------------------------------------- #

ax.set_title(f"{split_type} Graph Complexity: Nodes vs Links per Sample")
ax.set_xlabel("Number of Nodes")
ax.set_ylabel("Number of Links")
save_fig(fig, f"{split_type}_nodes_vs_links_scatter.png")


print("\nAll plots saved to:", plots_output_dir.resolve())