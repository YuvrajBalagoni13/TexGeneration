import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image


class ShaderDataset(Dataset):
    def __init__(
            self,
            dataset_dir: str = "Dataset",
            tokenizer_and_processor: any = None,
            max_seq_length : int = 2048
    ) -> None:
        super().__init__()
        self.samples = []
        self.dataset_path = Path(dataset_dir)
        self.processor = tokenizer_and_processor
        self.max_seq_length = max_seq_length
        
        # getting a list of dictionaries for all image & text shader pair
        for style_dir in self.dataset_path.iterdir():
            if style_dir.is_dir():
                images = list(style_dir.glob("*.jpg"))
                for image_path in images:
                    shader_path = image_path.with_suffix(".txt")
                    if shader_path.exists():
                        self.samples.append({
                            "image": image_path,
                            "shader": shader_path
                        })
        
        if not self.samples:
            raise RuntimeError("No valid image-shader pairs found.")
        
        print(f"--- Dataset Initialized: {len(self.samples)} pairs found ---")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        
        # loading image & text
        image = Image.open(sample["image"]).convert("RGB")
        with open(sample["shader"], "r") as f:
            shader_text = f.read()

        # main conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": (
                     "Generate a text based shader graph in the following format -\n"
                     "N|node_name:node_type;...\n"
                     "P|node_name.property_path:value;...\n"
                     "L|node_name.output_socket>node_name.input_socket;...\n"
                     "Here N| represents nodes, P| tells properties & L| tells links."
                     )},
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": shader_text}
                ]
            }
        ] 

        # Entire conversation
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
            truncation = True,
            max_length = self.max_seq_length,
            padding = False
        )

        prompt_inputs = self.processor(
            text = prompt_only,
            images = image,
            return_tensors = "pt",
            truncation = True,
            max_length = self.max_seq_length,
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
