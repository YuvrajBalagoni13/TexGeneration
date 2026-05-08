from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
import torch.nn as nn
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from tqdm.auto import tqdm

from dataset import ShaderDataset

# class TrainModel(nn.Module): 
#     def __init__(self) -> None:
#         super().__init__()
#         self.model = None
#         self.tokenizer = None 
#         self.device = None

#     def load_model(
#             self,
#             model_path: str = "unsloth/Qwen3.5-2B",
#             load_in_4bit: bool = True,
#             lora: bool = True,
#             lora_layers: list[str] = ['vision', 'language'],
#             max_seq_length: int = 2048
#     ) -> None:
        
#         self.model, self.tokenizer = FastVisionModel.from_pretrained(
#             model_name=model_path,
#             load_in_4bit=load_in_4bit,
#             use_gradient_checkpointing="unsloth",
#             max_seq_length=max_seq_length,
#         )

#         if lora:
#             self.model = FastVisionModel.get_peft_model(
#                 self.model, 
#                 finetune_vision_layers = "vision" in lora_layers,
#                 finetune_language_layers = "language" in lora_layers,
#                 finetune_attention_modules = "attention" in lora_layers,
#                 finetune_mlp_modules = "mlp" in lora_layers,
#                 r = 16,
#                 lora_alpha = 16,
#                 lora_dropout = 0,
#                 bias = "none",
#                 random_state = 3697,
#                 use_rslora = True,
#             )

device = "cuda" if torch.cuda.is_available() else "cpu"

model, processor = FastVisionModel.from_pretrained(
    model_name = "unsloth/Qwen3.5-2B",
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",
    max_seq_length = 2048
)

# for param in model.parameters():
    # param.requires_grad = False
# 
# model = FastVisionModel.get_peft_model(
    # model, 
    # finetune_vision_layers = True,
    # finetune_language_layers = True,
    # finetune_attention_modules = True,
    # finetune_mlp_modules = True,
    # r = 16,
    # lora_alpha = 16,
    # lora_dropout = 0,
    # bias = "none",
    # random_state = 3697,
    # use_rslora = True,
# )
# 
# model.print_trainable_parameters()

training_dataset = ShaderDataset("Dataset/train", processor)
testing_dataset = ShaderDataset("Dataset/infigen", processor)

print(training_dataset[0])
# training_dataloader = DataLoader(training_dataset, batch_size=4, shuffle=True, collate_fn=UnslothVisionDataCollator(model, processor))
# testing_dataloader = DataLoader(testing_dataset, batch_size=4, shuffle=False, collate_fn=UnslothVisionDataCollator(model, processor))

# model_optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
# criterion = nn.CrossEntropyLoss()
# gradient_scaler = GradScaler(enabled=(device == 'cuda'))

# total_epochs = 3

# for epoch in range(total_epochs):
#     model.train()
#     loss = 0
    
#     for batch_idx, current_batch in tqdm(enumerate(training_dataloader)):
#         token_ids = current_batch['input_ids'].to(device)
#         mask_for_attention = current_batch['attention_mask'].to(device)
#         pixel_data = current_batch['pixel_values'].to(device)
#         grid_info = current_batch['image_grid_thw'].to(device)
