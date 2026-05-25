import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm


class ShaderDataset(Dataset):
    def __init__(
            self,
            dataset_dir: str = "Dataset",
            tokenizer_and_processor: any = None,
            max_seq_length: int = 768,
            skip_over_length: bool = True,
    ) -> None:
        super().__init__()
        self.samples = []
        self.dataset_path = Path(dataset_dir)
        self.processor = tokenizer_and_processor
        self.max_seq_length = max_seq_length
        self.skip_over_length = skip_over_length

        skipped = 0
        all_pairs = []

        # collect all image-shader pairs
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

        # filter by length if skip_over_length is True
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

        # load image and shader text
        image = Image.open(sample["image"]).convert("RGB")
        with open(sample["shader"], "r") as f:
            shader_text = f.read()

        # conversation template
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

        # full conversation text (user + assistant)
        full_text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )

        # prompt only (user side) — for masking labels
        prompt_only = self.processor.apply_chat_template(
            conversation[:-1],
            tokenize=False,
            add_generation_prompt=True
        )

        # encode full conversation with image — only ONE image encoding
        inputs = self.processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
            padding=False
        )

        # get prompt token length WITHOUT encoding image again
        prompt_token_len = self.processor.tokenizer(
            prompt_only,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        )["input_ids"].shape[1]

        # build labels — mask prompt tokens with -100
        input_ids = inputs["input_ids"].squeeze(0)
        labels = input_ids.clone()
        labels[:prompt_token_len] = -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # squeeze all tensors and build result
        result = {k: v.squeeze(0) for k, v in inputs.items()}
        result["labels"] = labels

        # safety: add mm_token_type_ids if missing
        if "mm_token_type_ids" not in result:
            result["mm_token_type_ids"] = torch.zeros_like(input_ids)

        return result