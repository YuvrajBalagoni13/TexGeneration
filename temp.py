import os
# import unsloth
from transformers import AutoProcessor
from transformers import Qwen3_5ForConditionalGeneration, Qwen3_5Config
from peft import LoraConfig, get_peft_model, PeftModel
# from unsloth import FastVisionModel
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

from .dataset import ShaderDataset

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


#################################
#     log & save functions      #
#################################

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

def main(
        run_name = "Qwen3.5_0.8B_run_2.0",
        epochs = 5,
        batch_size = 2,
        lr = 1e-5,
        lora = True,
        lora_r = 32,
        lora_alpha = 64,
        gradient_accumulation = 8,
        load_ckpt_dir = "",
        load_state_dir = "",
        add_new_tokens = False,
        tokens_json_path = "",
        seed = 42

) -> None:
    seed_everything(seed)
    wandb.init(project="TexGeneration", name=run_name, config = {
        "epochs" : epochs,
        "batch_size" : batch_size,
        "lr" : lr,
        "lora-r" : lora_r,
        "lora-alpha" : lora_alpha,
        "gradient accumulation" : gradient_accumulation
    })

    ########################################
    #     Unsloth Model & lora Loading     #
    ########################################

    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # precision_type = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    # print(f"precision type : {precision_type}")
    # model, processor = FastVisionModel.from_pretrained(
    #    model_name = "unsloth/Qwen3.5-0.8B",
    #    load_in_4bit = False,
    #    use_gradient_checkpointing = True,
    #    max_seq_length = 1024,
    #   #  dtype = precision_type
    # )
    # model_precision_type = next(model.parameters()).dtype
    # print(f"Model dtype: {model_precision_type}") 

    ########################################
    #      custom Model & lora Loading     #
    ########################################

    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision_type = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    # precision_type = torch.float16
    print(f"precision type : {precision_type}")

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3.5-0.8B",
        torch_dtype=precision_type,
        device_map=device
    )
    processor = AutoProcessor.from_pretrained(
      "Qwen/Qwen3.5-0.8B"
    )

    model_precision_type = next(model.parameters()).dtype
    print(f"Model dtype: {model_precision_type}")

    for param in model.parameters():
      param.requires_grad_(False)
    ##########################
    #     Adding tokens      #
    ##########################
    if add_new_tokens:
        with open(tokens_json_path, "r") as f:
            tokens_dict = json.load(f)

        old_vocab_size = len(processor.tokenizer)

        tokens_list = tokens_dict["new_tokens"] + tokens_dict["special_tokens"]
        subwords_id_list = []
        for token in tokens_list:
            subwords = processor.tokenizer.tokenize(token)
            subwords_id = processor.tokenizer.convert_tokens_to_ids(subwords)
            subwords_id_list.append(subwords_id)

        processor.tokenizer.add_tokens(tokens_dict["new_tokens"])
        processor.tokenizer.add_special_tokens({
            "additional_special_tokens" : tokens_dict["special_tokens"]
        })
        model.resize_token_embeddings(len(processor.tokenizer))

        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()

        with torch.no_grad():
            for i, token in enumerate(tokens_list):
                token_id = processor.tokenizer.convert_tokens_to_ids(token)
                mean_in = input_embeddings.weight[subwords_id_list[i]].mean(dim=0)
                mean_out = output_embeddings.weight[subwords_id_list[i]].mean(dim=0)

                input_embeddings.weight[token_id] = mean_in
                output_embeddings.weight[token_id] = mean_out

        def freeze_old_input_grads(grad):
            grad[:old_vocab_size] = 0
            return grad
        
        def freeze_old_output_grads(grad):
            grad[:old_vocab_size] = 0
            return grad
        
        input_embeddings.requires_grad_(True)
        output_embeddings.requires_grad_(True)

        input_embeddings.weight.register_hook(freeze_old_input_grads)
        output_embeddings.weight.register_hook(freeze_old_input_grads)

        input_embeddings.to(model_precision_type)
        output_embeddings.to(model_precision_type)
        # embeddings = model.get_input_embeddings().weight.data
        # new_tokens_all = tokens_dict["new_tokens"] + tokens_dict["special_tokens"]
        
        # for token in new_tokens_all:
        #     token_id = processor.tokenizer.convert_tokens_to_ids(token)
        #     subwords = processor.tokenizer.tokenize(token)
        #     subwords_id = processor.tokenizer.convert_tokens_to_ids(subwords)
        #     if subwords_id:
        #         avg = embeddings[subwords_id].mean(dim=0)
        #         embeddings[token_id] = avg
            
        # def freeze_old_embeddings_hook(grad):
        #     grad[:old_vocab_length] = 0
        #     return grad
        
        # model.get_input_embeddings().weight.requires_grad_(True)
        # model.get_input_embeddings().weight.register_hook(freeze_old_embeddings_hook)

        print(f"Old vocab size: {old_vocab_size}")
        print(f"New vocab size: {len(processor.tokenizer)}")
        print(f"Added {len(processor.tokenizer) - old_vocab_size} new tokens")

    #########################
    #     lora Loading      #
    #########################

    # if lora:
    #     model = FastVisionModel.get_peft_model(
    #        model, 
    #        finetune_vision_layers = True,
    #        finetune_language_layers = True,
    #        finetune_attention_modules = True,
    #        finetune_mlp_modules = True,
    #        r = lora_r,
    #        lora_alpha = lora_alpha,
    #        lora_dropout = 0,
    #        bias = "none",
    #        random_state = 3697,
    #        use_rslora = True,
    #     ).to(device)

    # model.print_trainable_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} || Total: {total:,} || {100 * trainable / total:.2f}%")

    ############################
    #     Dataset Loading      #
    ############################
    training_dataset = ShaderDataset("/content/drive/MyDrive/ShaderDataset/train", processor, max_seq_length=1024)
    testing_dataset = ShaderDataset("/content/drive/MyDrive/ShaderDataset/val", processor, max_seq_length=1024)

    # this fills the pad_token_id because DataLoader only give batch as input to this so we fill this with ourselve before
    collate_fn = partial(shader_collate_fn, pad_token_id = processor.tokenizer.pad_token_id)

    generator = torch.Generator()
    generator.manual_seed(seed)
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, generator=generator)
    testing_dataloader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, generator=generator)
    
     ####################
    #     Training      #
    #####################
    if add_new_tokens:
        model_optimizer = torch.optim.AdamW(
          [p for p in model.parameters() if p.requires_grad],
          lr = lr
        )
    else:
        model_optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.gradient_checkpointing_enable()
    scaler = GradScaler()

    start_epoch = 0
    start_batch_idx = 0
    if load_ckpt_dir and load_state_dir:
        model, processor, model_optimizer, start_epoch, start_batch_idx = load_checkpoint(model, processor, model_optimizer, load_ckpt_dir, load_state_dir)

    total_epochs = epochs
    ACCUMULATION_INTERVAL = gradient_accumulation

    int_keys = {"input_ids", "attention_mask", "labels", "image_grid_thw", "mm_token_type_ids"}
    for epoch in range(start_epoch, total_epochs):
        model.train()
        loss = 0
        model_optimizer.zero_grad()
        for batch_idx, current_batch in tqdm(enumerate(training_dataloader)):

            if batch_idx < start_batch_idx:
                continue

            batch = {k : v.to(device)
                    for k, v in current_batch.items()}

            with autocast('dtype=torch.float16, enabled=True):
                outputs = model(**batch)
                batch_loss = outputs.loss

                batch_loss = batch_loss / ACCUMULATION_INTERVAL
            scaler.scale(batch_loss).backward()

            if (batch_idx + 1) % ACCUMULATION_INTERVAL == 0:
                scaler.unscale_(model_optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
                if torch.isnan(total_norm) or torch.isinf(total_norm):
                    print(f"Got NAN Gradients, Skipping step ...")
                    scaler.update()
                    model_optimizer.zero_grad()
                    continue
                scaler.step(model_optimizer)
                scaler.update() 
                # model_optimizer.step()
                model_optimizer.zero_grad()

            loss += batch_loss.item() * ACCUMULATION_INTERVAL

            if batch_idx % 5 == 0:
                log_metrics(epoch=epoch, iteration=batch_idx, loss=batch_loss.item() * ACCUMULATION_INTERVAL)

            if batch_idx % 250 == 0 and batch_idx != 0:
                save_checkpoint(epoch, batch_idx, run_name, model, processor, model_optimizer, True)

        loss = loss / len(training_dataloader)
        print(f"total loss - {loss} after epochs - {total_epochs}")

        #######################
        #     Evaluation      #
        #######################
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for eval_batch in tqdm(testing_dataloader):
                batch = {k : v.to(precision_type).to(device) if k not in int_keys else v.to(device)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    gradient_accumulation = 8,
    load_ckpt_dir = "",
    load_state_dir = ""
    parser.add_argument(
        "--run_name",
        type=str,
        default="Qwen3.5_0.8B_run_2.0"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        required=True
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        required=True
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        required=True
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=32
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=8,
        required=True
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
    parser.add_argument(
        "--add_new_tokens",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--tokens_json_path",
        type=str,
        default=""
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    args = parser.parse_args()

    main(
        run_name = args.run_name,
        epochs = args.epochs,
        batch_size = args.batch_size,
        lr = args.lr,
        lora = args.lora,
        lora_r = args.lora_r,
        lora_alpha = args.lora_alpha,
        gradient_accumulation = args.gradient_accumulation,
        load_ckpt_dir = args.load_ckpt_dir,
        load_state_dir = args.load_state_dir,
        add_new_tokens = args.add_new_tokens,
        tokens_json_path = args.tokens_json_path,
        seed = args.seed
    )

"""
python -m TexGeneration.src.model.train \
--run_name Qwen3.5_0.8B_run_2.2 \
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