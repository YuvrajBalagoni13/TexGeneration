import torch
from tqdm.auto import tqdm
import json
from pathlib import Path
from PIL import Image

from .inference import Inference

def load_eval_data(eval_data_path: str = "") -> list[dict]:
    image_path_list = list(Path(eval_data_path).rglob("*.jpg"))
    data = []
    for image_path in image_path_list:
        sample = {}
        sample["image"] = str(image_path)
        with open(image_path.with_suffix(".txt"), "r") as f:
            sample["shader"] = f.read()
        data.append(sample)
    print(f"---------- Total eval data length = {len(data)} samples ----------")
    return data

def main(
        data_path: str = "",
        model_path: str = "",
        save_response_path: str = ""
        ) -> None:
    # load eval data
    eval_data = load_eval_data(eval_data_path=data_path)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    
    # infer with the model
    inference_model = Inference(
        model_name = model_path,
        load_in_4bit = True,
        device = device
    )

    input_prompt = ("Generate a text based shader graph in the following format -\n"
                    "N|node_name:node_type;...\n"
                    "P|node_name.property_path:value;...\n"
                    "L|node_name.output_socket>node_name.input_socket;...\n"
                    "Here N| represents nodes, P| tells properties & L| tells links.")
    
    responses = {}

    for i, sample in tqdm(enumerate(eval_data)):
        sample_report = {}
        sample_report["image_path"] = str(sample["image"])
        sample_report["actual_shader"] = sample["shader"]

        output_txt_shader = inference_model.infer(
            image_path=sample["image"],
            input_prompt=input_prompt
        )

        sample_report["output_shader"] = output_txt_shader
        responses["sample_{i}"] = sample_report
    # save them in a json for validation & scoring

    with open(save_response_path, "w") as f:
        json.dump(responses, f, indent=4)

if __name__ == "__main__":
    data_path = "Dataset/eval/train_eval"
    model_path = "model_ckpts/run_0.1_lora"
    save_response_path = "Dataset/eval/train_response_run_0.1_response.json"
    main(
        data_path=data_path,
        model_path=model_path,
        save_response_path=save_response_path
        )
    
"""
python -m src.model.inference_pass
"""