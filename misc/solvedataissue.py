import unsloth
import os
import torch
import torch.nn as nn
import wandb
import argparse
import random
import numpy as np
import json
import torch.nn.functional as F
import yaml

from transformers import AutoProcessor
from transformers import Qwen3_5ForConditionalGeneration, Qwen3_5Config
from peft import LoraConfig, get_peft_model, PeftModel
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from functools import partial
from torch.amp import autocast, GradScaler
from unsloth import FastVisionModel
from transformers import get_cosine_schedule_with_warmup
from typing import Optional, List, Dict

from .dataset import ShaderDataset
from .embeddings import NewTokenEmbeddings, NewTokenOutput, new_tokens
from .utils import log_metrics, save_checkpoint, load_checkpoint
from .model import load_model, RegressionHead, TexGenModel

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def chunked_cross_entropy(logits, labels, chunk_size=512):
    total_loss = 0.0
    total_valid = 0
    seq_len = logits.size(1)
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_logits = logits[:, start:end, :]
        chunk_labels = labels[:, start:end]
        valid = (chunk_labels != -100).sum()
        if valid == 0:
            continue
        loss = F.cross_entropy(
            chunk_logits.reshape(-1, chunk_logits.size(-1)),
            chunk_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        total_loss += loss
        total_valid += valid
    return total_loss / total_valid.clamp(min=1)

def chunked_lm_loss(hidden_states, labels, lm_head, chunk_size=512):
    """Projects hidden_states -> logits chunk by chunk, avoiding ever
    materializing the full (batch, seq_len, vocab) tensor at once."""
    shift_hidden = hidden_states[:, :-1, :]
    shift_labels = labels[:, 1:]

    total_loss = 0.0
    total_valid = 0
    seq_len = shift_hidden.size(1)

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_hidden = shift_hidden[:, start:end, :]      # small: (batch, chunk_size, hidden_dim)
        chunk_labels = shift_labels[:, start:end]

        valid = (chunk_labels != -100).sum()
        if valid == 0:
            continue

        chunk_logits = lm_head(chunk_hidden)               # only THIS chunk gets projected to vocab size
        loss = F.cross_entropy(
            chunk_logits.reshape(-1, chunk_logits.size(-1)),
            chunk_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        total_loss += loss
        total_valid += valid

    return total_loss / total_valid.clamp(min=1)

def main(
        run_name = "Qwen3.5_0.8B_run_2.0",
        config_yaml = "",
        load_ckpt_dir = "",
        load_state_dir = ""
) -> None:
    # Load Config ------------------------------------------ #
    with open(config_yaml, "r") as f:
        config = yaml.safe_load(f)

    seed_everything(config['seed'])
    
   # Model Loading (Unsloth) + LoRA Initialization ----------------------------- #

    device = "cuda" if torch.cuda.is_available() else "cpu"

    precision_type = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    # precision_type = torch.float16

    model, processor = load_model(
        model_name = config['model_base'],
        quantize = config['quantize'],
        max_seq_len = config['max_seq_length'],
        lora = config['lora'],
        lora_layers = {
            "vision": config['vision'],
            "language": config['language'],
            "attention": config['attention'],
            "mlp": config['mlp']
        },
        lora_r = config['lora_r'],
        lora_alpha = config['lora_alpha'],
        lora_dropout = config['lora_dropout'],
        rslora = config['rslora'],
        precision_type = precision_type,
        device = device
    )
    if config['add_regression_head']:
        print(f"models hidden embed dim : {model.config.text_config.hidden_size}")
        regression_head = RegressionHead(
            embed_dim = model.config.text_config.hidden_size,
            hidden_dim = 512,
            dropout = 0.01
        ).to(device)

        model = TexGenModel(
            model = model,
            regression_head = regression_head,
            trigger_token_id = 248077
        )

    # Get Input Embeddings & LM Head --------------------- #

    if config['add_new_tokens']:
        model.model = new_tokens(
            model = model.model,
            token_json_path = config['tokens_json_path'], 
            processor = processor, 
            mean_subwords = config['mean_subwords'], 
            untie = config['untie'], 
            trainable = config['tokens_trainable'],
            precision_type = precision_type,
            device = device
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} || Total: {total:,} || {100 * trainable / total:.2f}%")

    # Dataset Loading ------------------------- #

    training_dataset = ShaderDataset(
        "/content/ShaderDataset/train", 
        processor, 
        max_seq_length=config['max_output_tokens'], 
        skip_over_length=config['skip_over_length'], 
        add_space_btw_nums=config['add_space_btw_nums'], 
        format_for_regression=config['add_regression_head']
    )
    testing_dataset = ShaderDataset(
        "/content/ShaderDataset/val", 
        processor, 
        max_seq_length=config['max_output_tokens'], 
        skip_over_length=config['skip_over_length'], 
        add_space_btw_nums=config['add_space_btw_nums'], 
        format_for_regression=config['add_regression_head']
    )

    collate_fn = partial(training_dataset.shader_collate_fn, pad_token_id = processor.tokenizer.pad_token_id)

    generator = torch.Generator()
    generator.manual_seed(config['seed'])
    training_dataloader = DataLoader(training_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn, generator=generator, num_workers=2, pin_memory=True)
    testing_dataloader = DataLoader(testing_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    
    # Optimizer & scheduler loading ------------------------------- #
    if config['add_new_tokens']:
        embedding_params = []
        model_params = []
        for name, params in model.named_parameters():
            if params.requires_grad:
                if ("new_embeddings" in name or "new_lm_head" in name):
                    embedding_params.append(params)
                else:
                    model_params.append(params)

        model_optimizer = torch.optim.AdamW([
            {"params": model_params, "lr": float(config['lr'])},
            {"params": embedding_params, "lr": float(config['lr_embeds'])}
        ],
        fused = True
        )
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        model_optimizer = torch.optim.AdamW(trainable_params, lr=float(config['lr']), fused=True)
    
    total_steps = 3000
    warmup_steps = config['warmup_steps']
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=model_optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    start_epoch = 0
    start_batch_idx = 0

    # Loading ckpt if given -------------------------------------------------- #
    if load_ckpt_dir and load_state_dir:
        model, processor, model_optimizer, scheduler, start_epoch, start_batch_idx = load_checkpoint(model, processor, model_optimizer, scheduler, load_ckpt_dir, load_state_dir, new_tokens=config['add_new_tokens'], regression_model=config['add_regression_head'])

    total_epochs = config['epochs']
    ACCUMULATION_INTERVAL = config['gradient_accumulation']

    criterion = nn.MSELoss()
    # Training Loop -------------------------------------------------- #
    wandb.init(project="TexGeneration", name=config['run_name'], config = config)
    for epoch in range(start_epoch, total_epochs):
        
        loss = 0
        model_optimizer.zero_grad()
        for batch_idx, current_batch in tqdm(enumerate(training_dataloader)):

            if batch_idx < start_batch_idx:
                continue
            
            model.train()
            batch = {k : v.to(device)
                    for k, v in current_batch['text'].items()}
            num_vals = current_batch['nums'].to(device)
            num_vals = torch.sign(num_vals) * torch.log1p(torch.abs(num_vals))

            batch_mse_loss = 0.0
            with autocast('cuda', dtype=precision_type):
                if config['add_regression_head']:
                    hidden_states, regression_preds = model(batch)
                    lm_head = model.model.get_output_embeddings()  # your NewTokenOutput module, or resized embedding
                else:
                    outputs = model(**batch)
                    vlm_logits = outputs.logits
                labels = batch["labels"]
      
                # shift_logits = vlm_logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                batch_loss = chunked_lm_loss(hidden_states, labels, lm_head, chunk_size=128)
                # batch_loss = chunked_cross_entropy(shift_logits, shift_labels, chunk_size=128)
                # batch_loss = F.cross_entropy(
                #     shift_logits.view(-1, shift_logits.size(-1)),
                #     shift_labels.view(-1),
                #     ignore_index=-100,
                # )
                batch_loss = batch_loss / ACCUMULATION_INTERVAL

                if config['add_regression_head'] and regression_preds is not None:
                    batch_mse_loss = criterion(regression_preds, num_vals)
                    batch_mse_loss = batch_mse_loss / ACCUMULATION_INTERVAL
                else:
                    batch_mse_loss = torch.tensor(0.0, device=batch_loss.device)

            final_loss = config['ce_lambda'] * batch_loss + config['mse_lambda'] * batch_mse_loss
            final_loss.backward()

            if (batch_idx + 1) % ACCUMULATION_INTERVAL == 0:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
                if torch.isnan(total_norm) or torch.isinf(total_norm):
                    print(f"Gradients having issue, Skipping step ...")
                    model_optimizer.zero_grad()
                model_optimizer.step()
                scheduler.step()
                model_optimizer.zero_grad()
                
            
            loss += final_loss.item() * ACCUMULATION_INTERVAL

            if batch_idx % 5 == 0:
                log_metrics(epoch=epoch, iteration=batch_idx, loss=batch_loss.item() * ACCUMULATION_INTERVAL, mse_loss = batch_mse_loss * ACCUMULATION_INTERVAL, lr = scheduler.get_last_lr()[0])

            if batch_idx % 250 == 0 and batch_idx != 0:
                save_checkpoint(epoch, batch_idx, wandb.run.name, model, processor, model_optimizer, scheduler, True, new_tokens=config['add_new_tokens'], regression_model=config['add_regression_head'])

            # -- Evaluation Loop ---------------------------------------------- #
            if batch_idx % 2500 == 0 and batch_idx != 0:
                model.eval()
                eval_loss = 0
                with torch.no_grad():
                    for eval_idx, eval_batch in enumerate(tqdm(testing_dataloader)):
                        if eval_idx > 50:
                            break
                        batch = {k : v.to(device)
                                for k, v in eval_batch['text'].items()}
                        num_vals = eval_batch['nums']

                        eval_mse_loss = 0.0
                        with autocast('cuda', dtype = precision_type):
                            if config["add_regression_head"]:
                                eval_outputs, eval_regression_preds = model(**batch)
                                eval_mse_loss = criterion(eval_regression_preds, num_vals)
                            eval_outputs = model(**batch)
                            eval_batch_loss = eval_outputs.loss
                            
                        eval_loss += eval_batch_loss.item()

                        if eval_idx % 5 == 0:
                            log_metrics(epoch=epoch, iteration=eval_idx, loss=eval_batch_loss.item(), mse_loss = eval_mse_loss, train=False)

                
        loss = loss / len(training_dataloader)
        print(f"total loss - {loss} after epochs - {total_epochs}")
        save_checkpoint(epoch, 0, wandb.run.name, model, processor, model_optimizer, True, new_tokens=config['add_new_tokens'], regression_model=config['add_regression_head'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name",
        type=str,
        default="Qwen3.5_0.8B_run_2.0"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=""
    )
    parser.add_argument(
        "--load_ckpt_dir",
        type=str,
        default=""
    )
    parser.add_argument(
        "--load_state_dir",
        type=str,
        default=""
    )
    args = parser.parse_args()

    main(
        run_name = args.run_name,
        config_yaml = args.config,
        load_ckpt_dir = args.load_ckpt_dir,
        load_state_dir = args.load_state_dir,
    )

"""
python -m TexGeneration.src.model.train \
--run_name Qwen3.5_0.8B_run_2.2 \
--quantize \
--mean_subwords \
--epochs 5 \
--batch_size 2 \
--lr 1e-5 \
--lora \
--lora_r 32 \
--lora_alpha 64 \
--gradient_accumulation 8 \
--add_new_tokens \
--tokens_json_path addition_tokens.json
"""