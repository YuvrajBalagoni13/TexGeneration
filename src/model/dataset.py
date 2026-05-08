import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Dataloader
from pathlib import Path
from PIL import Image


class ShaderDataset(Dataset):
    def __init__(
            self,
            dataset_dir: str = "Dataset",
            tokenizer_and_processor: any = None
    ) -> None:
        super().__init__()
        self.samples = []
        self.dataset_path = Path(dataset_dir)
        self.processor = tokenizer_and_processor
        
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
        
        image = Image.open(sample["image"]).convert("RGB")
        with open(sample["shader"], "r") as f:
            shader_text = f.read()

        prompt_structure = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Generate a text based shader graph..."},
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": shader_text}
                ]
            }
        ] 

        text = self.processor.apply_chat_template(prompt_structure, tokenize=False, add_generation_prompt=False)
        
        inputs = self.processor(
            text=text, 
            images=image, 
            padding=True, 
            return_tensors="pt"
        )

        return {k: v.squeeze(0) for k, v in inputs.items()}