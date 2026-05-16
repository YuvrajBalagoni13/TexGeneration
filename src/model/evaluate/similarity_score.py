import torch
from tqdm.auto import tqdm
import json
from pathlib import Path
import lpips
import torchvision.transforms as T
from PIL import Image
import argparse 

def load_image_tensor(path: str = "") -> torch.Tensor:
    try:
        img = Image.open(path).convert("RGB").resize((224, 224))
        tensor = T.ToTensor()(img).unsqueeze(0)
        tensor = tensor * 2 - 1 # normalizing between [-1, 1] for lpips
        return tensor
    except Exception as e:
        raise ValueError(f"Error - {e}")
    
def similarity_score(
        image_path: str = "",
        render_path: str = "",
        lpips: any = None,
        device: str = None
        ) -> float:
    image_tensor = load_image_tensor(image_path).to(device)
    render_tensor = load_image_tensor(render_path).to(device)

    score = lpips(image_tensor, render_tensor).item()
    return score

def main(
        eval_json_path: str = ""
) -> None:
    with open(eval_json_path, "r") as f:
        responses = json.load(f)

    device = "cpu"
    lpips_loss = lpips.LPIPS(net = 'vgg').to(device)

    print("----- scoring -----")
    avg_score = 0.0
    for k, v in responses.items():
        image_path = v["image_path"]
        render_path = v["render_path"]

        score = similarity_score(image_path=image_path, render_path=render_path, lpips=lpips_loss, device=device)
        responses[k]["score"] = float(score)
        avg_score = avg_score + score
    print("----- Done -----")

    avg_score = avg_score / len(responses)
    responses["avg_score"] = avg_score
    print(f"average score - {avg_score}")

    with open(eval_json_path, "w") as f:
        json.dump(responses, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save_json_path",
        required=True
    )

    args = parser.parse_args()
    
    main(args.save_json_path)