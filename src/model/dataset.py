import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
import json
import re


class ShaderDataset(Dataset):
    def __init__(
            self,
            dataset_dir: str = "Dataset",
            tokenizer_and_processor: any = None,
            max_seq_length : int = 2048,
            skip_over_length : bool = False,
            sample_json_path : str = None,
            train : bool = True,
            add_space_btw_nums: bool = False
    ) -> None:
        super().__init__()
        self.samples = []
        self.dataset_path = Path(dataset_dir)
        self.processor = tokenizer_and_processor
        self.max_seq_length = max_seq_length
        self.skip_over_length = skip_over_length
        self.add_space_btw_nums = add_space_btw_nums

        skipped = 0
        all_pairs = []

        if sample_json_path:
            with open(sample_json_path, "r") as f:
                sample_dict = json.load(f)

            samples_list = sample_dict['train' if train else 'val']
            for sample in samples_list:
                self.samples.append({
                    'image' : Path(sample['image'][1:-1]),
                    'shader' : Path(sample['shader'][1:-1])
                })
            print(f"--- Dataset Initialized: {len(self.samples)} pairs found ---")

        else:
            for style_dir in self.dataset_path.iterdir():
                if style_dir.is_dir():
                    for image_path in style_dir.rglob("*.jpg"):
                        shader_path = image_path.with_suffix(".txt")
                        if shader_path.exists():
                            all_pairs.append({
                                "image":  image_path,
                                "shader": shader_path
                            })

            if not all_pairs:
                raise RuntimeError(f"No valid image-shader pairs found in {dataset_dir}")

            if skip_over_length:
                for pair in tqdm(all_pairs, desc="Filtering by length"):
                    with open(pair["shader"], "r") as f:
                        shader_text = f.read()
                    token_len = len(self.processor.tokenizer.tokenize(shader_text))
                    if token_len <= max_seq_length:
                        self.samples.append(pair)
                    else:
                        skipped += 1
            else:
                self.samples = all_pairs

            print(f"--- Dataset Initialized: {len(self.samples)} pairs found | {skipped} skipped (over {max_seq_length} tokens) ---")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        
        image = Image.open(sample["image"]).convert("RGB")
        with open(sample["shader"], "r") as f:
            shader_text = f.read()
            shader_text = self.space_btw_nums(shader_text) if self.add_space_btw_nums else shader_text

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Generate a text based shader graph based on the given input image"},
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": shader_text}
                ]
            }
        ] 

        full_text = self.processor.apply_chat_template(
            conversation,
            tokenize = False,
            add_generation_prompt = False
        )

        # Only User side of the conversation
        prompt_only = self.processor.apply_chat_template(
            conversation[:-1],
            tokenize = False,
            add_generation_prompt = True
        )

        inputs = self.processor(
            text = full_text,
            images = image,
            return_tensors = "pt",
            padding = False
        )

        prompt_inputs = self.processor(
            text = prompt_only,
            images = image,
            return_tensors = "pt",
            padding = False
        )

        # masking user part of convo & padding tokens for labels 
        input_ids = inputs["input_ids"].squeeze(0)
        prompt_length = prompt_inputs["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_length] = -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        result = {k: v.squeeze(0) for k, v in inputs.items()}
        result["labels"] = labels

        return result

    @staticmethod
    def space_btw_nums(text: str) -> str:
        def space_digits(match):
            num_str = match.group(0)
            return " ".join(list(num_str))
        return re.sub(r'-?\d+\.?\d*', space_digits, text)
    
    @staticmethod
    def shader_collate_fn(batch, pad_token_id = 0):
        """
        Adds padding to all the values to stack the batches together.
        """
        input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_token_id)
        attention_mask = pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
        labels = pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=-100)
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
    
        result = {
            "input_ids" : input_ids,
            "attention_mask" : attention_mask,
            "pixel_values" : pixel_values,
            "labels" : labels,
        }
    
        if "image_grid_thw" in batch[0]:
            result["image_grid_thw"] = torch.stack([b["image_grid_thw"] for b in batch])

        if "mm_token_type_ids" in batch[0]:
            result["mm_token_type_ids"] = pad_sequence([b["mm_token_type_ids"] for b in batch], batch_first=True, padding_value=0)
        
        if "spatial_shapes" in batch[0]:
            result["spatial_shapes"] = torch.stack([b["spatial_shapes"] for b in batch])
            
        if "pixel_attention_mask" in batch[0]:
            result["pixel_attention_mask"] = pad_sequence([b["pixel_attention_mask"] for b in batch], batch_first=True, padding_value=0)
        return result
    


