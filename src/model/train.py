import os
import unsloth
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig
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

def shader_collate_fn(batch, pad_token_id = 0):
    """
    Adds padding to all the values to stack the batches together.
    """
    input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_token_id)
    attention_mask = pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
    mm_token_type_ids = pad_sequence([b["mm_token_type_ids"] for b in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=-100)

    pixel_values = torch.stack([b["pixel_values"] for b in batch])

    result = {
        "input_ids" : input_ids,
        "attention_mask" : attention_mask,
        "mm_token_type_ids" : mm_token_type_ids,
        "pixel_values" : pixel_values,
        "labels" : labels,
    }

    if "image_grid_thw" in batch[0]:
        result["image_grid_thw"] = torch.stack([b["image_grid_thw"] for b in batch])
    
    return result


#################################
#     log & save functions      #
#################################

wandb.init(project="TexGeneration", name="run_0.1", config = {
    "epochs" : 5,
    "batch_size" : 2,
    "lr" : 5e-5
})

def log_metrics(epoch, iteration, loss):
    print(f"epoch {epoch + 1} | iteration {iteration} | train loss - {loss:.2f}")
    wandb.log({
        "epoch" : epoch,
        "iteration" : iteration,
        "train loss" : loss
    })

def save_checkpoint(epoch, iteration, run_name, model, processor, log_wandb):
    print(f"------ Saving model checkpoint for epoch {epoch + 1} & iteration {iteration} ------")
    checkpoint_directory = f"./texgen_{run_name}_{epoch + 1}_{iteration}"
    os.makedirs(checkpoint_directory, exist_ok=True)
    model.save_pretrained(checkpoint_directory)
    processor.save_pretrained(checkpoint_directory)

    if log_wandb:
        artifact = wandb.Artifact(
            name=f"texgen_lora_{run_name}",
            type="model",
            description=f"LoRA adapter weights - epoch {epoch+1} iteration {iteration}",
            metadata={
                "epoch": epoch + 1,
                "iteration": iteration,
                "run_name": run_name,
            }
        )
        artifact.add_dir(checkpoint_directory)  
        wandb.log_artifact(artifact)
    print(f"✅ Model for epoch {epoch+1} & {iteration} saved to {checkpoint_directory}")


########################################
#     Unsloth Model & lora Loading     #
########################################

device = "cuda" if torch.cuda.is_available() else "cpu"

precision_type = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

model, processor = FastVisionModel.from_pretrained(
   model_name = "unsloth/Qwen3.5-2B",
   load_in_4bit = True,
   use_gradient_checkpointing = False,
   max_seq_length = 2048,
   dtype = precision_type
)

# for param in model.parameters():
#     param.requires_grad = False

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
).to(device)

model.print_trainable_parameters()

#############################################
#     Transformers Model & lora Loading     #
#############################################

# device = "cuda" if torch.cuda.is_available() else "cpu"
# 
# precision_type = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
# 
# model_name = "Qwen/Qwen3.5-2B"
# 
# bnb_config = BitsAndBytesConfig(
    # load_in_4bit = True,
    # bnb_4bit_compute_dtype = precision_type,
    # bnb_4bit_use_double_quant = True,
    # bnb_4bit_quant_type = "nf4"
# )
# model = Qwen3_5ForConditionalGeneration.from_pretrained(
    # model_name,
    # torch_dtype=precision_type,
    # device_map="auto",
    # quantization_config = bnb_config
# ).to(device)
# processor = AutoProcessor.from_pretrained(model_name)
# 
# for param in model.parameters():
    # param.requires_grad = False
# 
# model.gradient_checkpointing_enable()
# 
# lora_config = LoraConfig(
    # r=16,
    # lora_alpha=32,
    # target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    # lora_dropout=0.1,
    # bias="none",
    # use_rslora=True,
# )
# 
# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()

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

total_epochs = 5
ACCUMULATION_INTERVAL = 4


for epoch in range(total_epochs):
    model.train()
    loss = 0
    progress_bar = tqdm(training_dataloader, leave = True)
    model_optimizer.zero_grad()
    for batch_idx, current_batch in enumerate(progress_bar):
        batch = {k : v.to(precision_type).to(device) if v.dtype == torch.float32 else v.to(device)
                for k, v in current_batch.items()}

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

        if batch_idx % 5 == 0:
            log_metrics(epoch=epoch, iteration=batch_idx, loss=batch_loss.item() * ACCUMULATION_INTERVAL)

    loss = loss / len(training_dataloader)
    print(f"total loss - {loss} after epochs - {total_epochs}")

    #######################
    #     Evaluation      #
    #######################
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for eval_batch in tqdm(testing_dataloader):
            batch = {k : v.to(torch.float16).to(device) if v.dtype == torch.float32 else v.to(device)
                     for k, v in eval_batch.items()}
            
            eval_outputs = model(**batch)
            eval_batch_loss = eval_outputs.loss

            eval_loss += eval_batch_loss.item()

        wandb.log({
            "epoch" : epoch,
            "eval loss" : eval_loss / len(testing_dataloader)
        })

        print(f"Epoch {epoch} | evaluation loss - {eval_loss}")

    save_checkpoint(epoch, 0, wandb.run.name, model, processor, True)
            
