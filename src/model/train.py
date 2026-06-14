import unsloth
import os
from transformers import AutoProcessor
from transformers import Qwen3_5ForConditionalGeneration, Qwen3_5Config
from peft import LoraConfig, get_peft_model, PeftModel
import torch.nn as nn
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from functools import partial
import wandb
import argparse
import random
import numpy as np
import json
from torch.amp import autocast, GradScaler
from unsloth import FastVisionModel
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
import yaml

from .dataset import ShaderDataset
from .embeddings import NewTokenEmbeddings, NewTokenOutput
from .utils import log_metrics, save_checkpoint, load_checkpoint

# -- seed & collate functions ---------------------- #

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

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

def main(
        run_name = "Qwen3.5_0.8B_run_2.0",
        # quantize = False,
        # mean_subwords = True,
        # epochs = 5,
        # batch_size = 2,
        # lr = 1e-5,
        # lora = True,
        # lora_r = 32,
        # lora_alpha = 64,
        # gradient_accumulation = 8,
        config_yaml = "",
        load_ckpt_dir = "",
        load_state_dir = "",
        # add_new_tokens = False,
        # tokens_json_path = "",
        # seed = 42

) -> None:
    with open(config_yaml, "r") as f:
        config = yaml.safe_load(f)

    seed_everything(config['seed'])
    
   # -- Model Loading (Unsloth) ----------------------------- #

    device = "cuda" if torch.cuda.is_available() else "cpu"

    precision_type = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    # precision_type = torch.float16

    model, processor = FastVisionModel.from_pretrained(
       model_name = "unsloth/Qwen3.5-0.8B",
       load_in_4bit = config['quantize'],
       use_gradient_checkpointing = True,
       max_seq_length = config['max_seq_length'],
       dtype = precision_type
    )

    # -- LoRA Initialization ------------------------------ #

    model = FastVisionModel.get_peft_model(
       model, 
       finetune_vision_layers = True,
       finetune_language_layers = True,
       finetune_attention_modules = True,
       finetune_mlp_modules = True,
       r = config['lora_r'],
       lora_alpha = config['lora_alpha'],
       lora_dropout = config['lora_dropout'],
       bias = "none",
       random_state = 3697,
       use_rslora = config['rslora'],
    ).to(device)
    print("------- LoRA Trainable parameters -------")
    model.print_trainable_parameters()

    # -- Get Input Embeddings & LM Head --------------------- #

    if config['add_new_tokens']:
        with open(config['tokens_json_path'], "r") as f:
            tokens_dict = json.load(f)

        old_vocab_size = len(processor.tokenizer)

        new_tokens = tokens_dict["new_tokens"] + tokens_dict["special_tokens"]
        subwords_id_list = []
        for token in new_tokens:
            subwords = processor.tokenizer.tokenize(token)
            subwords_id = processor.tokenizer.convert_tokens_to_ids(subwords)
            subwords_id_list.append(subwords_id)
        
        processor.tokenizer.add_tokens(new_tokens)
        # processor.tokenizer.add_special_tokens({
        #     "additional_special_tokens" : tokens_dict["special_tokens"]
        # })

        # untying the weights
        if model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr():
          print("Weights are tied ...")
          model.lm_head.weight = nn.Parameter(
              model.get_output_embeddings().weight.clone()
          )
          model.lm_head.weight.requires_grad_(False)

        input_embeddings = model.get_input_embeddings()
        output_lm_head = model.get_output_embeddings()

        new_embedding_layer = NewTokenEmbeddings(
            old_embeddings = input_embeddings,
            old_vocab_size = old_vocab_size,
            embed_dim = 1024,
            tokenizer = processor.tokenizer,
            mean_subwords = config['mean_subwords'],
            subwords_id_list = subwords_id_list,
            new_tokens = new_tokens
        )
        new_lm_head = NewTokenOutput(
            old_lm_head = output_lm_head,
            embed_dim = 1024,
            old_vocab_size = old_vocab_size,
            tokenizer = processor.tokenizer,
            mean_subwords = config['mean_subwords'],
            subwords_id_list = subwords_id_list,
            new_tokens = new_tokens
        )

        # new_lm_head.new_lm_head.weight = new_embedding_layer.new_embeddings.weight

        new_vocab_size = len(processor.tokenizer)
        model.config.vocab_size = new_vocab_size
        model.config.text_config.vocab_size = new_vocab_size
        setattr(model.config, "vocab_size", new_vocab_size)

        model.set_input_embeddings(new_embedding_layer)
        model.set_output_embeddings(new_lm_head)

        # if model.get_input_embeddings().new_embeddings.weight.data_ptr() == model.get_output_embeddings().new_lm_head.weight.data_ptr():
        #     print(f"----- Successfully tied new embeddings & new lm head -----")

        new_embedding_layer.to(precision_type).to(device)
        new_lm_head.to(precision_type).to(device)

        # for saving & loading checkpoints
        config_class = model.config.__class__
        if not isinstance(getattr(config_class, "vocab_size", None), property):
            config_class.vocab_size = property(
                lambda self: self.text_config.vocab_size
            )
    
        model.get_input_embeddings().new_embeddings.requires_grad_(True)
        model.get_output_embeddings().new_lm_head.requires_grad_(True)

        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("------- Tokens Trainable parameters enabled & checkpointed -------")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} || Total: {total:,} || {100 * trainable / total:.2f}%")

    # -- Dataset Loading ------------------------- #

    training_dataset = ShaderDataset("/content/ShaderDataset/train", processor, max_seq_length=config['max_output_tokens'], skip_over_length=True, sample_json_path="/content/drive/MyDrive/ShaderDataset/dataset_samples.json", train=True)
    testing_dataset = ShaderDataset("/content/ShaderDataset/val", processor, max_seq_length=config['max_output_tokens'], skip_over_length=True, sample_json_path="/content/drive/MyDrive/ShaderDataset/dataset_samples.json", train=False)

    collate_fn = partial(shader_collate_fn, pad_token_id = processor.tokenizer.pad_token_id)

    generator = torch.Generator()
    generator.manual_seed(config['seed'])
    training_dataloader = DataLoader(training_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn, generator=generator, num_workers=2, pin_memory=True)
    testing_dataloader = DataLoader(testing_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn, generator=generator, num_workers=2, pin_memory=True)
    
    # -- Training ------------------------------- #
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
            {"params": embedding_params, "lr": float(config['lr_embeds'])},
            {"params": model_params, "lr": float(config['lr'])}
        ],
        fused = True
        )
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        model_optimizer = torch.optim.Adam(trainable_params, lr=config['lr'], fused=True)
    
    total_steps = 2500 
    warmup_steps = config['warmup_steps']
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=model_optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    start_epoch = 0
    start_batch_idx = -1
    if load_ckpt_dir and load_state_dir:
        model, processor, model_optimizer, scheduler, start_epoch, start_batch_idx = load_checkpoint(model, processor, model_optimizer, scheduler, load_ckpt_dir, load_state_dir)

    total_epochs = config['epochs']
    ACCUMULATION_INTERVAL = config['gradient_accumulation']

    wandb.init(project="TexGeneration", name=config['run_name'], config = config)

    for epoch in range(start_epoch, total_epochs):
        
        loss = 0
        model_optimizer.zero_grad()
        for batch_idx, current_batch in tqdm(enumerate(training_dataloader)):

            if batch_idx < start_batch_idx + 1:
                continue
            
            model.train()
            batch = {k : v.to(device)
                    for k, v in current_batch.items()}
            
            labels = batch.pop("labels")

            with autocast('cuda', dtype=precision_type):
                outputs = model(**batch)
                logits = outputs.logits

                shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
                shift_labels = labels[..., 1:].contiguous().view(-1)
                batch_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
                
                # batch_loss = outputs.loss
                batch_loss = batch_loss / ACCUMULATION_INTERVAL

            batch_loss.backward()

            if (batch_idx + 1) % ACCUMULATION_INTERVAL == 0:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
                if torch.isnan(total_norm) or torch.isinf(total_norm):
                    print(f"Gradients having issue, Skipping step ...")
                    model_optimizer.zero_grad()
                model_optimizer.step()
                scheduler.step()
                model_optimizer.zero_grad()
                
            
            loss += batch_loss.item() * ACCUMULATION_INTERVAL

            if batch_idx % 5 == 0:
                log_metrics(epoch=epoch, iteration=batch_idx, loss=batch_loss.item() * ACCUMULATION_INTERVAL, lr = scheduler.get_last_lr()[1])

            if batch_idx % 250 == 0 and batch_idx != 0:
                save_checkpoint(epoch, batch_idx, run_name, model, processor, model_optimizer, scheduler, True)

            # -- Evaluation ------------------------ #
            if batch_idx % 2500 == 0 and batch_idx != 0:
                model.eval()
                eval_loss = 0
                with torch.no_grad():
                    for eval_idx, eval_batch in enumerate(tqdm(testing_dataloader)):
                        if eval_idx > 50:
                            break
                        batch = {k : v.to(device)
                                for k, v in eval_batch.items()}

                        labels = batch.pop("labels")

                        with autocast('cuda', dtype = precision_type):
                            eval_outputs = model(**batch)
                            logits = eval_outputs.logits

                            shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
                            shift_labels = labels[..., 1:].contiguous().view(-1)
                            eval_batch_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

                        eval_loss += eval_batch_loss.item()

                        if eval_idx % 5 == 0:
                            log_metrics(epoch=epoch, iteration=eval_idx, loss=eval_batch_loss.item(), train=False)

                
        loss = loss / len(training_dataloader)
        print(f"total loss - {loss} after epochs - {total_epochs}")
        save_checkpoint(epoch, 0, wandb.run.name, model, processor, model_optimizer, True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name",
        type=str,
        default="Qwen3.5_0.8B_run_2.0"
    )
    parser.add_argument(
        "--config_yaml",
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
        config_yaml = args.config_yaml,
        # quantize = args.quantize,
        # mean_subwords = args.mean_subwords,
        # epochs = args.epochs,
        # batch_size = args.batch_size,
        # lr = args.lr,
        # lora = args.lora,
        # lora_r = args.lora_r,
        # lora_alpha = args.lora_alpha,
        # gradient_accumulation = args.gradient_accumulation,
        load_ckpt_dir = args.load_ckpt_dir,
        load_state_dir = args.load_state_dir,
        # add_new_tokens = args.add_new_tokens,
        # tokens_json_path = args.tokens_json_path,
        # seed = args.seed
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