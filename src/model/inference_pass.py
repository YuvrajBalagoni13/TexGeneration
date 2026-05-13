# python src/model/inference_pass.py
import json
import torch
from pathlib import Path
from .inference import Inference
from ..data.dataset import load_eval_data

def run_inference(data_path, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = Inference(
        model_name="model_ckpts/run_0.1_lora",
        load_in_4bit=True,
        device=device
    )
    
    input_prompt = (
        "Generate a text based shader graph in the following format -\n"
        "N|node_name:node_type;...\n"
        "P|node_name.property_path:value;...\n"
        "L|node_name.output_socket>node_name.input_socket;...\n"
        "Here N| represents nodes, P| tells properties & L| tells links."
    )

    eval_data = load_eval_data(data_path)
    results = []

    for sample in eval_data:
        output_shader = model.infer(
            image_path=sample["image"],
            input_prompt=input_prompt
        )
        results.append({
            "image_path": str(sample["image"]),
            "actual_shader": sample["shader"],
            "output_shader": output_shader
        })

    # save results to JSON for blender to pick up
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Inference done. Saved to {output_path}")

if __name__ == "__main__":
    run_inference("Dataset/eval/train_eval", "Dataset/eval/inference_results.json")