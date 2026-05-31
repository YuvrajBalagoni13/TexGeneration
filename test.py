import wandb
api = wandb.Api()
artifact = api.artifact("yuvrajbalagoni-indian-institute-of-technology-dhanbad/TexGeneration/texgen_lora_LoRA_token_main:v12")
artifact_dir = artifact.download()