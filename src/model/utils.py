import os
import torch
import wandb

from peft import PeftModel, get_peft_model
from transformers import AutoProcessor
from unsloth import FastVisionModel
from typing import Optional, Dict, List

def load_optimizer():
    return

def log_metrics(
        epoch : int = None, 
        iteration : int = None, 
        loss : float = None, 
        lr : float = None, 
        train : bool = True):
    if train:
        print(f"epoch {epoch + 1} | iteration {iteration} | lr - {lr} | train loss - {loss:.2f} ")
        wandb.log({
            "epoch" : epoch,
            "iteration" : iteration,
            "train loss" : loss,
            "lr" : lr
        })
    else:
        print(f"epoch {epoch + 1} | eval iteration {iteration} | eval loss - {loss:.2f} ")
        wandb.log({
            "epoch" : epoch,
            "eval iteration" : iteration,
            "eval loss" : loss
        })
        

def save_checkpoint(epoch, iteration, run_name, model, processor, optimizer, scheduler, log_wandb, new_tokens):
    print(f"------ Saving model checkpoint for epoch {epoch + 1} & iteration {iteration} ------")
    checkpoint_directory = f"./ckpts_{run_name}_{epoch + 1}_{iteration}/texgen_{run_name}_{epoch + 1}_{iteration}"
    resume_directory = f"./ckpts_{run_name}_{epoch + 1}_{iteration}/texgen_{run_name}_{epoch + 1}_{iteration}_state"
    os.makedirs(checkpoint_directory, exist_ok=True)
    model.save_pretrained(checkpoint_directory, save_embedding_layers=False)
    processor.save_pretrained(checkpoint_directory)
    
    if new_tokens:
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
    torch.save(scheduler.state_dict(), os.path.join(resume_directory, "scheduler.pth"))

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

def load_checkpoint(base_model, processor, optimizer, scheduler, checkpoint_directory, optimizer_directory, new_tokens):
    print(f"------ Loading checkpoint from {checkpoint_directory} ------")
    
    base_model.load_adapter(checkpoint_directory, adapter_name='default')
    processor = AutoProcessor.from_pretrained(checkpoint_directory)
    
    if new_tokens:
        base_model.get_input_embeddings().new_embeddings.load_state_dict(
            torch.load(os.path.join(checkpoint_directory, "new_embeddings.pth"))
        )
        base_model.get_output_embeddings().new_lm_head.load_state_dict(
            torch.load(os.path.join(checkpoint_directory, "new_lm_head.pth"))
        )
    optimizer.load_state_dict(
        torch.load(os.path.join(optimizer_directory, "optimizer.pth"),
        map_location='cuda')  
    )
    scheduler.load_state_dict(
        torch.load(os.path.join(optimizer_directory, "scheduler.pth"),
        map_location='cuda')
    )
    state = torch.load(os.path.join(optimizer_directory, "training_state.pth"))
    epoch = state['epoch']
    iteration = state['iteration']
    
    print(f"------ Resumed from epoch {epoch + 1}, iteration {iteration} ------")
    return base_model, processor, optimizer, scheduler, epoch, iteration
