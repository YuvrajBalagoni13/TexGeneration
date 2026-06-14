import torch
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
from peft import PeftModel
from pathlib import Path

def main(
      model_ckpt : str,
      save_path : str | Path
):
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3.5-0.8B",
        torch_dtype = torch.float16,
        device_map = 'cuda'
    )
    processor = AutoProcessor.from_pretrained(model_ckpt)
    len(processor.tokenizer)

    model = PeftModel.from_pretrained(model, model_ckpt)
    model = model.merge_and_unload()

    new_embeds = torch.load(model_ckpt / 'new_embeddings.pth', map_location='cuda')['weight']
    new_lm_heads = torch.load(model_ckpt / 'new_lm_head.pth', map_location='cuda')['weight']

    old_embed_len = 248077
    n = new_embeds.shape[0]

    model.resize_token_embeddings(len(processor.tokenizer), pad_to_multiple_of=128)

    if model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr():
      print("Weights are tied, untying ...")
      model.lm_head.weight = torch.nn.Parameter(
          model.get_output_embeddings().weight.clone()
      )
      model.lm_head.weight.requires_grad_(False)
      print("Weights Untied")

    old_embeddings = model.get_input_embeddings()
    old_lm_head = model.get_output_embeddings()

    old_embeddings.weight[old_embed_len : old_embed_len + n] = new_embeds
    old_lm_head.weight[old_embed_len : old_embed_len + n] = new_lm_heads

    model.set_input_embeddings(old_embeddings)
    model.set_output_embeddings(old_lm_head)

    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f"---------- Saved model at {save_path} ----------")

if __name__ == "__main__":
   main(
      model_ckpt="",
      save_path=""
   )