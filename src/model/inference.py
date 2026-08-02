# from unsloth import FastVisionModel
import torch
import json
import random
import argparse
import re

from tqdm import tqdm
from typing import Optional, Any
from peft import PeftModel
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
from PIL import Image
from pathlib import Path

from .model import RegressionHead, TexGenModel
from .utils import load_checkpoint
from .dataset import ShaderDataset

class Inference:
    def __init__(
            self,
            model_base: str = None,
            ckpt_path: str = None,
            quantize: bool = True,
            max_seq_length : int = 450,
            device: str = None,
            precision_type: Optional[Any] = None,
            new_tokens: bool = False,
            regression_model: bool = True
    ) -> None:
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if precision_type:
            self.precision_type = precision_type
        else:
            self.precision_type = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        self.max_seq_length = max_seq_length

        self.message = [
            {
                "role" : "user",
                "content" : [
                    {"type" : "image"},
                    {"type" : "text", "text" : "Generate a text based shader graph based on the given input image"}
                ]
            }
        ]

        if regression_model:
            vlm_model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_base,
                torch_dtype = self.precision_type,
                device_map = self.device
            )
            regression_head = RegressionHead(
              embed_dim = vlm_model.config.text_config.hidden_size,
              hidden_dim = 512,
              dropout = 0.01
            ).to(self.precision_type).to(self.device)

            self.model = TexGenModel(
                model = vlm_model,
                regression_head = regression_head,
                trigger_token_id = 248077
            )
        else:
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_base,
                torch_dtype = self.precision_type,
                device_map = self.device
            )
                
        self.processor = AutoProcessor.from_pretrained(ckpt_path)
        

        if ckpt_path:
            self.model, self.processor, _, _, _, _ = load_checkpoint(
                base_model = self.model, 
                processor = self.processor, 
                checkpoint_directory = ckpt_path,  
                regression_model = regression_model
            )
        
        self.processor.tokenizer.padding_side = "left"

        if new_tokens:
            new_embeddings = torch.load(Path(ckpt_path) / "new_embeddings.pth", map_location=self.device)

            if isinstance(new_embeddings, dict):
                new_embeddings = new_embeddings[list(new_embeddings.keys())[0]]

            self.model.model.resize_token_embeddings(len(self.processor.tokenizer), pad_to_multiple_of=None)

            n = new_embeddings.shape[0]
            self.model.model.get_input_embeddings().weight.data[-n:] = new_embeddings.to(self.precision_type)
        
        in_emb = self.model.model.get_input_embeddings()
        out_emb = self.model.model.get_output_embeddings()
        
        print("tied:", in_emb.weight.data_ptr() == out_emb.weight.data_ptr())
        print(self.model.model.active_adapters())
        print(self.model.model.peft_config)

        vocab_size = len(self.processor.tokenizer)
        last_n_ids = list(range(vocab_size - n, vocab_size))
        print(self.processor.tokenizer.convert_ids_to_tokens(last_n_ids))
        
        print("final embedding size:", self.model.model.get_input_embeddings().weight.shape[0])
        print("tokenizer size:", len(self.processor.tokenizer))
        self.model.model.eval()

    def infer(
            self,
            image_paths : list[str] = None,
            prompt : str = None
    ) -> dict:
        images = [
            Image.open(image) for image in image_paths
        ]
        texts = [
            self.processor.apply_chat_template(
                self.message,
                tokenize = False,
                add_generation_prompt = True
            ) for _ in image_paths
        ]
        inputs = self.processor(
            text = texts,
            images = images,
            return_tensors = "pt",
        ).to(self.device)

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.precision_type)

        eos_ids = list(set(filter(None, [
            self.processor.tokenizer.eos_token_id,
            self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
        ])))

        with torch.no_grad():
            outputs_list, regression_preds_list = self.model.generate(
                inputs,
                max_new_tokens = self.max_seq_length,
                do_sample = False,
                temperature = 0.3,
                top_p = 0.95,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=eos_ids
            )
        
        results = {}
        for i, new_tokens in enumerate(outputs_list):
            decoded = self.processor.decode(new_tokens, skip_special_tokens=False)
            preds = regression_preds_list[i]
            preds = preds.cpu().float().numpy().tolist()
            preds = [f"{pred:.3f}" for pred in preds] 

            decoded = decoded.replace("<|im_end|>", "")
            decoded = decoded.replace("<|endoftext|>", "")
            preds_iter = iter(preds)
            final_shader_text = re.sub(r"<NUM>", lambda match: next(preds_iter), decoded)
            results[str(image_paths[i])] = final_shader_text
        return results
            
def main(
        model_base : str,
        lora_path : str,
        eval_data_path : str,
        quantize : bool,
        data_length : int,
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
        ckpt_path=lora_path,
        quantize=quantize,
        max_seq_length=512,
        new_tokens=new_tokens,
        regression_model=regression_model
    )

    batches = [dataset_list[i:i + batch_size] for i in range(0, data_length, batch_size)]
    print(f"----- Processing {len(batches)} batches with {batch_size} batch sizes (total samples = {data_length}) -----")
    for batch in tqdm(batches):
        outputs = inference.infer(batch)
        for k, v in outputs.items():
            results[k] = v

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
        batch_size=args.batch_size,
        new_tokens=args.new_tokens,
        regression_model=args.regression_model
    )

    with open(args.save_json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"----- Saved results in {args.save_json_path} -----")

"""
python -m src.model.inference \
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