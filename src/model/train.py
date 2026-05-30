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

from .dataset import ShaderDataset
# from .shader_dataset import ShaderDataset

# -- Custom Embedding classes ---------------------- #

class NewTokenEmbeddings(nn.Module):
    def __init__(
            self, 
            old_embeddings : nn.Embedding = None,
            embed_dim : int = 1024,
            old_vocab_size : int = 248077,
            tokenizer : any = None,
            mean_subwords: bool = False,
            subwords_id_list : list = None,
            new_tokens : list = None
        ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.old_embeddings = old_embeddings # [248077, 1024]
        self.old_vocab_size = old_vocab_size
        self.old_embeddings.requires_grad_(False)

        self.num_new_tokens = len(new_tokens)
        self.new_embeddings = nn.Embedding(self.num_new_tokens, embed_dim) # [237, 1024]
        print(f"old vocab size - {self.old_vocab_size}")
        print(f"new token size - {self.num_new_tokens}")

        with torch.no_grad():
            if mean_subwords:
                if not subwords_id_list:
                    raise ValueError(f"not subwords id list given ...")
                for i, token in enumerate(new_tokens):
                    token_id = tokenizer.convert_tokens_to_ids(token)
                    avg = self.old_embeddings.weight[subwords_id_list[i]].mean(dim=0)
                    self.new_embeddings.weight[token_id - self.old_vocab_size] = avg
            else:
                avg = self.old_embeddings.weight.mean(dim=0)
                for i, token in enumerate(new_tokens):
                    token_id = tokenizer.convert_tokens_to_ids(token)
                    self.new_embeddings.weight[token_id - self.old_vocab_size] = avg

    def forward(self, input_ids):
        old_ids = torch.clamp(input_ids, min=0, max=self.old_vocab_size - 1)
        new_ids = torch.clamp(input_ids - self.old_vocab_size, min=0, max=self.num_new_tokens - 1)

        old_vectors = self.old_embeddings(old_ids)
        new_vectors = self.new_embeddings(new_ids)

        is_old = (input_ids < self.old_vocab_size).unsqueeze(-1).to(old_vectors.dtype) 

        return old_vectors * is_old + new_vectors * (1.0 - is_old)
    
class NewTokenOutput(nn.Module):
    def __init__(
            self, 
            old_lm_head : nn.Linear = None,
            embed_dim : int = 1024,
            old_vocab_size : int = 248077,
            tokenizer : any = None,
            mean_subwords: bool = False,
            subwords_id_list : list = None,
            new_tokens : list = None
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.old_lm_head = old_lm_head
        self.old_vocab_size = old_vocab_size
        self.old_lm_head.requires_grad_(False)

        self.num_new_tokens = len(new_tokens)
        self.new_lm_head = nn.Linear(embed_dim, self.num_new_tokens, bias = False)

        with torch.no_grad():
            if mean_subwords:
                if not subwords_id_list:
                    raise ValueError(f"not subwords id list given ...")
                for i, token in enumerate(new_tokens):
                    token_id = tokenizer.convert_tokens_to_ids(token)
                    avg = self.old_lm_head.weight[subwords_id_list[i]].mean(dim=0)
                    self.new_lm_head.weight[token_id - self.old_vocab_size] = avg
            else:
                avg = self.old_lm_head.weight.mean(dim=0)
                for i, token in enumerate(new_tokens):
                    token_id = tokenizer.convert_tokens_to_ids(token)
                    self.new_lm_head.weight[token_id - self.old_vocab_size] = avg
            
    @property
    def weight(self):
        return self.new_lm_head.weight

    def forward(self, hidden_states):
        # hidden_states - [batch_size, seq_len, 1024] 
        target_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.old_lm_head.weight.dtype)
        old_token_logits = self.old_lm_head(hidden_states).to(target_dtype)  # [batch_size, seq_len, 248077]
        old_token_logits = old_token_logits[..., :self.old_vocab_size]
        new_token_logits = self.new_lm_head(hidden_states).to(target_dtype)  # [batch_size, seq_len, 233]
        logits = torch.cat([old_token_logits, new_token_logits], dim=-1) # [batch_size, seq_len, 248310]
        return logits

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

# -- log & save functions ---------------------- #

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
    
    torch.save(
        model.get_input_embeddings().new_embeddings.state_dict(),
        os.path.join(checkpoint_directory, "new_embeddings.pth")
    )
    torch.save(
        model.get_output_embeddings().new_lm_head.state_dict(),
        os.path.join(checkpoint_directory, "new_lm_head.pth")
    )
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
    
    model.get_input_embeddings().new_embeddings.load_state_dict(
        torch.load(os.path.join(checkpoint_directory, "new_embeddings.pth"))
    )
    model.get_output_embeddings().new_lm_head.load_state_dict(
        torch.load(os.path.join(checkpoint_directory, "new_lm_head.pth"))
    )
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
        quantize = False,
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
    
   # -- Model Loading (Unsloth) ----------------------------- #

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # precision_type = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    precision_type = torch.float16

    model, processor = FastVisionModel.from_pretrained(
       model_name = "unsloth/Qwen3.5-0.8B",
       load_in_4bit = quantize,
       use_gradient_checkpointing = True,
       max_seq_length = 768,
       dtype = precision_type
    )

    # -- Get Input Embeddings & LM Head --------------------- #

    if add_new_tokens:
        with open(tokens_json_path, "r") as f:
            tokens_dict = json.load(f)

        old_vocab_size = len(processor.tokenizer)

        new_tokens = tokens_dict["new_tokens"] + tokens_dict["special_tokens"]
        subwords_id_list = []
        for token in new_tokens:
            subwords = processor.tokenizer.tokenize(token)
            subwords_id = processor.tokenizer.convert_tokens_to_ids(subwords)
            subwords_id_list.append(subwords_id)
        
        processor.tokenizer.add_tokens(tokens_dict["new_tokens"])
        processor.tokenizer.add_special_tokens({
            "additional_special_tokens" : tokens_dict["special_tokens"]
        })

        # untying the weights
        if model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr():
          print("Weights are tied ...")
          model.lm_head.weight = nn.Parameter(
              model.get_output_embeddings().weight.clone()
          )

        input_embeddings = model.get_input_embeddings()
        output_lm_head = model.get_output_embeddings()

        new_embedding_layer = NewTokenEmbeddings(
            old_embeddings = input_embeddings,
            old_vocab_size = old_vocab_size,
            embed_dim = 1024,
            tokenizer = processor.tokenizer,
            mean_subwords = True,
            subwords_id_list = subwords_id_list,
            new_tokens = new_tokens
        )
        new_lm_head = NewTokenOutput(
            old_lm_head = output_lm_head,
            embed_dim = 1024,
            old_vocab_size = old_vocab_size,
            tokenizer = processor.tokenizer,
            mean_subwords = True,
            subwords_id_list = subwords_id_list,
            new_tokens = new_tokens
        )

        new_vocab_size = len(processor.tokenizer)
        model.config.vocab_size = new_vocab_size
        model.config.text_config.vocab_size = new_vocab_size
        setattr(model.config, "vocab_size", new_vocab_size)

        model.set_input_embeddings(new_embedding_layer)
        model.set_output_embeddings(new_lm_head)

        new_embedding_layer.to(precision_type).to(device)
        new_lm_head.to(precision_type).to(device)

        # for saving & loading checkpoints
        config_class = model.config.__class__
        if not isinstance(getattr(config_class, "vocab_size", None), property):
            config_class.vocab_size = property(
                lambda self: self.text_config.vocab_size
            )

    # -- LoRA Initialization ------------------------------ #

    if lora:
        model = FastVisionModel.get_peft_model(
           model, 
           finetune_vision_layers = True,
           finetune_language_layers = True,
           finetune_attention_modules = True,
           finetune_mlp_modules = True,
           r = lora_r,
           lora_alpha = lora_alpha,
           lora_dropout = 0,
           bias = "none",
           random_state = 3697,
           use_rslora = True,
        ).to(device)
        print("------- LoRA Trainable parameters -------")
        model.print_trainable_parameters()
    
    if add_new_tokens:
      model.get_input_embeddings().new_embeddings.requires_grad_(True)
      model.get_output_embeddings().new_lm_head.requires_grad_(True)

      model.enable_input_require_grads()
      model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
      print("------- Tokens Trainable parameters enabled & checkpointed -------")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} || Total: {total:,} || {100 * trainable / total:.2f}%")

    # -- Dataset Loading ------------------------- #

    # using 768 as max seq length because p95 of data distribution is 751 - can refer to token_analysis.png
    training_dataset = ShaderDataset("/content/ShaderDataset/train", processor, max_seq_length=768, skip_over_length=True)
    testing_dataset = ShaderDataset("/content/ShaderDataset/val", processor, max_seq_length=768, skip_over_length=True)

    collate_fn = partial(shader_collate_fn, pad_token_id = processor.tokenizer.pad_token_id)

    generator = torch.Generator()
    generator.manual_seed(seed)
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, generator=generator, num_workers=2, pin_memory=True)
    testing_dataloader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, generator=generator, num_workers=2, pin_memory=True)
    
    # -- Training ------------------------------- #
    if add_new_tokens:
        embedding_params = []
        model_params = []
        for name, params in model.named_parameters():
            if params.requires_grad:
                if ("new_embeddings" in name or "new_lm_head" in name):
                    embedding_params.append(params)
                else:
                    model_params.append(params)

        model_optimizer = torch.optim.AdamW([
            {"params": embedding_params, "lr": lr * 0.01},
            {"params": model_params, "lr": lr}
        ],
        fused = True
        )
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        model_optimizer = torch.optim.Adam(trainable_params, lr=lr, fused=True)

    start_epoch = 0
    start_batch_idx = 0
    if load_ckpt_dir and load_state_dir:
        model, processor, model_optimizer, start_epoch, start_batch_idx = load_checkpoint(model, processor, model_optimizer, load_ckpt_dir, load_state_dir)

    total_epochs = epochs
    ACCUMULATION_INTERVAL = gradient_accumulation

    wandb.init(project="TexGeneration", name=run_name, config = {
        "epochs" : epochs,
        "batch_size" : batch_size,
        "lr" : lr,
        "lora-r" : lora_r,
        "lora-alpha" : lora_alpha,
        "gradient accumulation" : gradient_accumulation
    })

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
                model_optimizer.zero_grad()

            loss += batch_loss.item() * ACCUMULATION_INTERVAL

            if batch_idx % 5 == 0:
                log_metrics(epoch=epoch, iteration=batch_idx, loss=batch_loss.item() * ACCUMULATION_INTERVAL)

            if batch_idx % 250 == 0 and batch_idx != 0:
                save_checkpoint(epoch, batch_idx, run_name, model, processor, model_optimizer, True)

        loss = loss / len(training_dataloader)
        print(f"total loss - {loss} after epochs - {total_epochs}")

        # -- Evaluation ------------------------ #

        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for eval_batch in tqdm(testing_dataloader):
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
        "--quantize",
        action="store_true",
        default=False
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
        quantize = args.quantize,
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
--quantize \
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