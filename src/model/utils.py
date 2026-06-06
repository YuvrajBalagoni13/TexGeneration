import os
import torch
from peft import PeftModel
from transformers import AutoProcessor
import wandb


def log_metrics(epoch, iteration, loss, lr):
    print(f"epoch {epoch + 1} | iteration {iteration} | lr - {lr} | train loss - {loss:.2f} ")
    wandb.log({
        "epoch" : epoch,
        "iteration" : iteration,
        "train loss" : loss,
        "lr" : lr
    })

def save_checkpoint(epoch, iteration, run_name, model, processor, optimizer, scheduler, log_wandb):
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

def load_checkpoint(base_model, processor, optimizer, scheduler, checkpoint_directory, optimizer_directory):
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
    scheduler.load_state_dict(
        torch.load(os.path.join(optimizer_directory, "scheduler.pth"),
        map_location='cuda')
    )
    state = torch.load(os.path.join(optimizer_directory, "training_state.pth"))
    epoch = state['epoch']
    iteration = state['iteration']
    
    print(f"------ Resumed from epoch {epoch + 1}, iteration {iteration} ------")
    return model, processor, optimizer, epoch, iteration
