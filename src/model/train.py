from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
import torch.nn as nn
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from functools import partial
import wandb

from .dataset import ShaderDataset

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
    
    return result


#################################
#     wandb initialization      #
#################################
wandb.init(project="TexGeneration", name="run_0.1", config = {
    "epochs" : 5,
    "batch_size" : 2
})

def log_metrics(epoch, iteration, loss):
    print(f"epoch {epoch + 1} | iteration {iteration} | train loss - {loss:.2f}")
    wandb.log({
        "epoch" : epoch,
        "iteration" : iteration,
        "train loss" : loss
    })


#################################
#     Model & lora Loading      #
#################################
device = "cuda" if torch.cuda.is_available() else "cpu"

model, processor = FastVisionModel.from_pretrained(
    model_name = "unsloth/Qwen3.5-2B",
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",
    max_seq_length = 2048
)

for param in model.parameters():
    param.requires_grad = False

model = FastVisionModel.get_peft_model(
    model, 
    finetune_vision_layers = True,
    finetune_language_layers = True,
    finetune_attention_modules = True,
    finetune_mlp_modules = True,
    r = 16,
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    random_state = 3697,
    use_rslora = True,
)

model.print_trainable_parameters()

############################
#     Dataset Loading      #
############################
training_dataset = ShaderDataset("Dataset/train", processor, max_seq_length=2048)
testing_dataset = ShaderDataset("Dataset/infinigen", processor, max_seq_length=2048)

# this fills the pad_token_id because DataLoader only give batch as input to this so we fill this with ourselve before
collate_fn = partial(shader_collate_fn, pad_token_id = processor.tokenizer.pad_token_id)

training_dataloader = DataLoader(training_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
testing_dataloader = DataLoader(testing_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
 
 ####################
#     Training      #
#####################
model_optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss(ignore_index = -100)
gradient_scaler = GradScaler(enabled=(device == 'cuda'))

total_epochs = 5
ACCUMULATION_INTERVAL = 4

for epoch in range(total_epochs):
    model.train()
    loss = 0
    progress_bar = tqdm(training_dataloader, leave = True)
    model_optimizer.zero_grad()
    for batch_idx, current_batch in enumerate(progress_bar):
        batch = {k : v.to(device) for k, v in current_batch.items()}

        outputs = model(**batch)
        batch_loss = outputs.loss

        batch_loss = batch_loss / ACCUMULATION_INTERVAL
        batch_loss.backward()

        if (batch_idx + 1) % ACCUMULATION_INTERVAL == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
            model_optimizer.step()
            model_optimizer.zero_grad()

        loss += batch_loss.item() * ACCUMULATION_INTERVAL
        progress_bar.set_postfix(loss = batch_loss.item() * ACCUMULATION_INTERVAL)

        if batch_idx % 10 == 0:
            log_metrics(epoch=epoch, iteration=batch_idx, loss=batch_loss.item() * ACCUMULATION_INTERVAL)

    loss = loss / len(training_dataloader)
    print(f"total loss - {loss} after epochs - {total_epochs}")

    #######################
    #     Evaluation      #
    #######################
    # model.eval()
    # eval_loss = 0
    # with torch.no_grad():
    #     for eval_batch in tqdm(testing_dataloader):
    #         eval_outputs = model(**eval_batch)
    #         eval_batch_loss = outputs.loss

    #         eval_loss += eval_batch_loss.item()
            