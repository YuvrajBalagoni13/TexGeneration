import wandb
api = wandb.Api()
artifact = api.artifact("yuvrajbalagoni-indian-institute-of-technology-dhanbad/TexGeneration/texgen_lora_LoRA_token_scheduler_0.1_scale_lr_embed_06:v9")
artifact_dir = artifact.download()
