import os
import torch
import wandb

from typing import Optional, Any
from transformers import AutoProcessor

class SymLogNorm:
    def norm(self, x):
        return torch.sign(x) * torch.log1p(torch.abs(x))
    def inv_norm(self, x):
        return torch.sign(x) * torch.expm1(torch.abs(x))

def log_metrics(
        epoch : int = None, 
        iteration : int = None, 
        loss : float = None, 
        mse_loss: float = None,
        lr : float = None, 
        train : bool = True):
    if train:
        print(f"epoch {epoch + 1} | iteration {iteration} | lr - {lr} | train loss - {loss:.2f} | mse loss - {mse_loss:.2f}")
        wandb.log({
            "epoch" : epoch,
            "iteration" : iteration,
            "train loss" : loss,
            "train mse loss": mse_loss,
            "lr" : lr
        })
    else:
        print(f"epoch {epoch + 1} | eval iteration {iteration} | eval loss - {loss:.2f} | mse loss - {mse_loss:.2f}")
        wandb.log({
            "epoch" : epoch,
            "eval iteration" : iteration,
            "eval loss" : loss,
            "eval mse loss": mse_loss
        })
        

def save_checkpoint(epoch, iteration, run_name, base_model, processor, optimizer, scheduler, log_wandb, new_tokens, regression_model):
    print(f"------ Saving model checkpoint for epoch {epoch + 1} & iteration {iteration} ------")
    checkpoint_directory = f"./ckpts_{run_name}_{epoch + 1}_{iteration}/texgen_{run_name}_{epoch + 1}_{iteration}"
    resume_directory = f"./ckpts_{run_name}_{epoch + 1}_{iteration}/texgen_{run_name}_{epoch + 1}_{iteration}_state"
    os.makedirs(checkpoint_directory, exist_ok=True)
    if not regression_model:
        base_model.save_pretrained(checkpoint_directory, save_embedding_layers=False)
    else:
        base_model.model.save_pretrained(checkpoint_directory, save_embedding_layers=False)
        torch.save(
            base_model.regression_head.state_dict(),
            os.path.join(checkpoint_directory, "regression_head.pth")
        )
    processor.save_pretrained(checkpoint_directory)
    
    if new_tokens:
        torch.save(
            base_model.get_input_embeddings().new_embeddings.state_dict() if not regression_model else base_model.model.get_input_embeddings().new_embeddings.state_dict(),
            os.path.join(checkpoint_directory, "new_embeddings.pth")
        )
        torch.save(
            base_model.get_output_embeddings().new_lm_head.state_dict() if not regression_model else base_model.model.get_output_embeddings().new_lm_head.state_dict(),
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

def load_checkpoint(
    base_model: Any, 
    processor: Any, 
    checkpoint_directory: str, 
    optimizer: Optional[Any] = None, 
    scheduler: Optional[Any] = None, 
    optimizer_directory: Optional[str] = None, 
    new_tokens: bool = False, 
    regression_model: bool = False
):
    print(f"------ Loading checkpoint from {checkpoint_directory} ------")
    
    if not regression_model:
        base_model.load_adapter(checkpoint_directory, adapter_name='default')
    else:
        base_model.model.load_adapter(checkpoint_directory, adapter_name='default')
        base_model.regression_head.load_state_dict(torch.load(os.path.join(checkpoint_directory, "regression_head.pth")))
    processor = AutoProcessor.from_pretrained(checkpoint_directory)
    
    if new_tokens:
        if regression_model:
            base_model.model.get_input_embeddings().new_embeddings.load_state_dict(
                torch.load(os.path.join(checkpoint_directory, "new_embeddings.pth"))
            )
            base_model.model.get_output_embeddings().new_lm_head.load_state_dict(
                torch.load(os.path.join(checkpoint_directory, "new_lm_head.pth"))
            )
        else:
            base_model.get_input_embeddings().new_embeddings.load_state_dict(
                torch.load(os.path.join(checkpoint_directory, "new_embeddings.pth"))
            )
            base_model.get_output_embeddings().new_lm_head.load_state_dict(
                torch.load(os.path.join(checkpoint_directory, "new_lm_head.pth"))
            )
    if optimizer_directory is not None:
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
    else:
        return base_model, processor, None, None, None, None
