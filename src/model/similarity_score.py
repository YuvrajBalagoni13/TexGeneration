from PIL import Image
import lpips 
import json
import torch
import torch.nn as nn
from pathlib import Path
from tqdm.auto import tqdm
import argparse
import torchvision.transforms as T

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

    score = lpips(image_tensor, render_tensor)
    score = torch.exp(-score).item()
    return score

def main(eval_result_path = "results_info_temp_0_3_top_p_0_95.json"):
    with open(eval_result_path, "r") as f:
        results = json.load(f)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    lpips_loss = lpips.LPIPS(net = 'vgg').to(device)

    print("----- Scoring -----")
    avg_score = 0.0
    no_of_errors = 0
    shader_count = 0
    for image, val in tqdm(results.items()):
        if val['output_error']:
            results[image]['LPIPS_score'] = 0.0
            no_of_errors += 1
            continue
        if val['shader_error']:
            shader_count += 1
            continue

        output_image = Path(val['render_output'])
        shader_image = Path(val['render_shader'])

        score = similarity_score(shader_image, output_image, lpips_loss, device)
        results[image]['score'] = score
        avg_score += score
    print("----- Done Scoring -----")
    
    avg_score /= (len(results) - shader_count - no_of_errors)
    results['metadata'] = {
        'avg_score' : avg_score,
        'no_of_errors' : no_of_errors
    }

    print(f"avg_score : {avg_score}")
    print(f"no of errors : {no_of_errors}")
    
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
python -m src.model.similarity_score \
--result_json_path json_path     
"""