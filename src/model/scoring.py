# src/model/scoring.py
import json
import torch
import lpips
from PIL import Image
import torchvision.transforms as T

def load_image_tensor(path, device):
    img = Image.open(path).convert("RGB").resize((224, 224))
    tensor = T.ToTensor()(img).unsqueeze(0) * 2 - 1
    return tensor.to(device)

def run_scoring(render_report_path, final_report_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    with open(render_report_path, "r") as f:
        results = json.load(f)

    avg_score = 0.0
    for sample in results:
        if sample["error"] or not sample["render_path"]:
            sample["score"] = 1.0
        else:
            sample["score"] = lpips_fn(
                load_image_tensor(sample["image_path"], device),
                load_image_tensor(sample["render_path"], device)
            ).item()
        avg_score += sample["score"]

    avg_score /= len(results)
    final = {"samples": results, "avg_score": avg_score}

    with open(final_report_path, "w") as f:
        json.dump(final, f, indent=4)
    print(f"Avg LPIPS score: {avg_score:.4f}")

if __name__ == "__main__":
    run_scoring(
        "Dataset/eval/render_report.json",
        "Dataset/eval/final_report.json"
    )