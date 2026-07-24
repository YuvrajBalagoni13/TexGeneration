import torch
import torch.nn as nn
from pathlib import Path
from peft import get_peft_model, PeftModel
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from PIL import Image
import argparse
import json 
from tqdm.auto import tqdm
import random

from .dataset import ShaderDataset
from .inference import Inference

def main(
        model_base : str,
        lora_path : str,
        eval_data_path : str,
        quantize : bool,
        data_length : int,
        batch_process : bool,
        batch_size : int,
        new_tokens : bool,
        regression_model: bool
) -> dict:
    # dataset_list = random.sample(list(Path(eval_data_path).rglob("*.jpg")), data_length)
    processor = AutoProcessor.from_pretrained(lora_path)
    dataset = ShaderDataset(eval_data_path, processor, max_seq_length=450, skip_over_length=False)
    sample_list = random.sample(dataset.samples, data_length)

    dataset_list = []
    results = {}
    for sample in sample_list:
      dataset_list.append(sample['image'])

    inference = Inference(
        model_base=model_base,
        lora_path=lora_path,
        quantize=quantize,
        max_seq_length=450,
        new_tokens=new_tokens,
        regression_model=regression_model
    )

    if batch_process:
        batches = [dataset_list[i:i + batch_size] for i in range(0, 100, batch_size)]
        print(f"----- Processing {len(batches)} batches with {batch_size} batch sizes (total samples = {data_length}) -----")
        for batch in tqdm(batches):
            outputs = inference.batch_infer(batch)

            for k, v in outputs.items():
                results[k] = v
    else:
        print(f"----- Processing {len(dataset_list)} images -----")
        for image in tqdm(dataset_list[:100]):
            output, num_preds = inference.infer(image)
            results[str(image)] = {
                "shader_text": output,
                "nums": num_preds
            }
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="base model weights path if saved or download base model"
    )
    parser.add_argument(
        "--eval_data_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--save_json_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--data_length",
        type=int,
        default=100
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--batch_process",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4
    )
    parser.add_argument(
        "--new_tokens",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--regression_model",
        action="store_true",
        default=False
    )
    args = parser.parse_args()

    results = main(
        model_base=args.base_model,
        lora_path=args.lora_path,
        eval_data_path=args.eval_data_path,
        data_length=args.data_length,
        quantize=args.quantize,
        batch_process=args.batch_process,
        batch_size=args.batch_size,
        new_tokens=args.new_tokens,
        regression_model=args.regression_model
    )

    with open(args.save_json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"----- Saved results in {args.save_json_path} -----")

"""
python -m src.model.infer \
--base_model Qwen/Qwen3.5-0.8B \
--lora_path lorapath \
--eval_data_path datapath \
--save_json_path jsonpath \
--data_length 1000 \
--quantize \
--batch_process \
--batch_size 4 \
--new_tokens
"""