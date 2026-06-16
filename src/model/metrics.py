from PIL import Image
import lpips 
import json
import torch
import torch.nn as nn
from pathlib import Path
from tqdm.auto import tqdm
import argparse
import torchvision.transforms as T
import clip



def load_image_tensor(path: str = "") -> torch.Tensor:
    try:
        img = Image.open(path).convert("RGB").resize((224, 224))
        tensor = T.ToTensor()(img).unsqueeze(0)
        tensor = tensor * 2 - 1 # normalizing between [-1, 1] for lpips
        return tensor
    except Exception as e:
        raise ValueError(f"Error - {e}")
    
class LPIPS:
    def __init__(self, device: str, exp : bool = False) -> None:
        self.lpips_loss = lpips.LPIPS(net = 'vgg').to(device)
        self.exp = exp
    
    def score(
            self,
            image_tensor: torch.Tensor,
            render_tensor: torch.Tensor
    ) -> float:
        score = self.lpips_loss(image_tensor, render_tensor)
        if self.exp:
            score = torch.exp(-score)
        return score.item()

class CLIP:
    def __init__(
            self,
            model : str = 'ViT-B/32',
            device : str = 'cpu'
        ) -> None:
        self.model, self.processor = clip.load(model, device=device)
        self.device = device
    
    def score(
            self,
            image_path: str,
            render_path: str
    ) -> float:
        image_input = self.processor(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
        render_input = self.processor(Image.open(render_path).convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            render_features = self.model.encode_image(render_input)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        render_features /= render_features.norm(dim=-1, keepdim=True)

        return (image_features @ render_features.T).item()

def main(eval_result_path = "results_info_temp_0_3_top_p_0_95.json"):
    with open(eval_result_path, "r") as f:
        results = json.load(f)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    
    lpips_scorer = LPIPS(device=device, exp=False)
    clip_scorer = CLIP(model='ViT-B/32', device=device)

    print("----- Scoring -----")
    avg_lpips_score = 0.0
    avg_clip_score = 0.0
    no_of_errors = 0
    shader_count = 0
    for image, val in tqdm(results.items()):
        if image == 'metadata':
            continue

        if val['output_error']:
            results[image]['LPIPS_score'] = 0.0
            no_of_errors += 1
            continue
        if val['shader_error']:
            shader_count += 1
            continue

        output_image = Path(val['render_output'])
        shader_image = Path(val['render_shader'])

        output_tensor = load_image_tensor(output_image).to(device)
        shader_tensor = load_image_tensor(shader_image).to(device)

        lpips_score = lpips_scorer.score(output_tensor, shader_tensor)
        clip_score = clip_scorer.score(output_image, shader_image)

        results[image]['lpips_score'] = lpips_score
        results[image]['clip_score'] = clip_score
        avg_lpips_score += lpips_score
        avg_clip_score += clip_score
    print("----- Done Scoring -----")
    
    avg_lpips_score /= (len(results) - shader_count - no_of_errors)
    avg_clip_score /= (len(results) - shader_count - no_of_errors)
    results['metadata'] = {
        'avg_lpips_score' : avg_lpips_score,
        'avg_clip_score' : avg_clip_score,
        'no_of_errors' : no_of_errors,
        'performance_model' : {
            'lpips_score' : avg_lpips_score * (1 + (no_of_errors / len(results))),
            'clip_score' : avg_clip_score * (1 - (no_of_errors / len(results)))
        }
    }

    print(f"avg lpips score : {avg_lpips_score}")
    print(f"avg clip score : {avg_clip_score}")
    print(f"no of errors : {no_of_errors}")
    print(f"Overall performance lpips score : {avg_lpips_score * (1 + (no_of_errors / len(results)))}")
    print(f"Overall performance clip score : {avg_clip_score * (1 - (no_of_errors / len(results)))}")
    
    with open(eval_result_path, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_json_path",
        type = str,
        required = True
    )
    args = parser.parse_args()
    main(eval_result_path=args.result_json_path)

"""
python -m src.model.metrics \
--result_json_path result_inference_json/results/results_untied_main_2.json
"""