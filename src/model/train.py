import os
import unsloth
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, PeftModel
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

def save_checkpoint(epoch, iteration, run_name, model, processor, optimizer, log_wandb):
    print(f"------ Saving model checkpoint for epoch {epoch + 1} & iteration {iteration} ------")
    checkpoint_directory = f"./ckpts_{run_name}_{epoch + 1}_{iteration}/texgen_{run_name}_{epoch + 1}_{iteration}"
    resume_directory = f"./ckpts_{run_name}_{epoch + 1}_{iteration}/texgen_{run_name}_{epoch + 1}_{iteration}_state"
    os.makedirs(checkpoint_directory, exist_ok=True)
    model.save_pretrained(checkpoint_directory)
    processor.save_pretrained(checkpoint_directory)
    os.makedirs(resume_directory, exist_ok=True)

    torch.save(optimizer.state_dict(), os.path.join(resume_directory, "optimizer.pth"))

    torch.save({
        'epoch': epoch,
        'iteration': iteration,
        'run_name': run_name,
    }, os.path.join(resume_directory, "training_state.pth"))

    print(f"------ Checkpoint saved to {checkpoint_directory} ------")
    print(f"------ Resume Checkpoint saved to {resume_directory} ------")

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
        artifact.add_dir(f"ckpts_{run_name}_{epoch + 1}_{iteration}")  
        wandb.log_artifact(artifact)
    print(f"✅ Model for epoch {epoch+1} & {iteration} saved to {checkpoint_directory}")

def load_checkpoint(base_model, processor, optimizer, checkpoint_directory, optimizer_directory):
    print(f"------ Loading checkpoint from {checkpoint_directory} ------")
    
    model = PeftModel.from_pretrained(base_model, checkpoint_directory)
    processor = AutoProcessor.from_pretrained(checkpoint_directory)
    
    optimizer.load_state_dict(
        torch.load(os.path.join(optimizer_directory, "optimizer.pth"),
        map_location='cuda')  
    )
    
    state = torch.load(os.path.join(optimizer_directory, "training_state.pth"))
    epoch = state['epoch']
    iteration = state['iteration']
    
    print(f"------ Resumed from epoch {epoch + 1}, iteration {iteration} ------")
    return model, processor, optimizer, epoch, iteration


########################################
#     Unsloth Model & lora Loading     #
########################################

device = "cuda" if torch.cuda.is_available() else "cpu"

precision_type = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

model, processor = FastVisionModel.from_pretrained(
   model_name = "unsloth/Qwen3.5-2B",
   load_in_4bit = True,
   use_gradient_checkpointing = False,
   max_seq_length = 1024,
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
training_dataset = ShaderDataset("/content/drive/MyDrive/ShaderDataset/train", processor, max_seq_length=1024)
testing_dataset = ShaderDataset("/content/drive/MyDrive/ShaderDataset/val", processor, max_seq_length=1024)

# this fills the pad_token_id because DataLoader only give batch as input to this so we fill this with ourselve before
collate_fn = partial(shader_collate_fn, pad_token_id = processor.tokenizer.pad_token_id)

training_dataloader = DataLoader(training_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
testing_dataloader = DataLoader(testing_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
 
 ####################
#     Training      #
#####################
model_optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

ckpt_dir = ""
state_dir = ""

start_epoch = 0
if ckpt_dir and state_dir:
    model, processor, model_optimizer, start_epoch, batch_idx = load_checkpoint(model, processor, model_optimizer, ckpt_dir, state_dir)

total_epochs = 5
ACCUMULATION_INTERVAL = 4


for epoch in range(start_epoch, total_epochs):
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

        if batch_idx % 250 == 0 and batch_idx != 0:
            save_checkpoint(epoch, batch_idx, wandb.run.name, model, processor, model_optimizer, True)

    loss = loss / len(training_dataloader)
    print(f"total loss - {loss} after epochs - {total_epochs}")

    #######################
    #     Evaluation      #
    #######################
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for eval_batch in tqdm(testing_dataloader):
            batch = {k : v.to(precision_type).to(device) if v.dtype == torch.float32 else v.to(device)
                     for k, v in eval_batch.items()}
            
            eval_outputs = model(**batch)
            eval_batch_loss = eval_outputs.loss

            eval_loss += eval_batch_loss.item()

        wandb.log({
            "epoch" : epoch,
            "eval loss" : eval_loss / len(testing_dataloader)
        })

        print(f"Epoch {epoch} | evaluation loss - {eval_loss}")

    save_checkpoint(epoch, 0, wandb.run.name, model, processor, model_optimizer, True)
            
