import torch
import torch.nn as nn

from unsloth import FastVisionModel
from typing import Optional, Dict

def load_model(
        model_name: str,
        quantize: bool,
        max_seq_len: int,
        precision_type: str,
        lora: bool = True,
        lora_layers: Optional[Dict[str, bool]] = None,
        lora_r: Optional[int] = None,
        lora_alpha: Optional[int] = None,
        lora_dropout: Optional[float] = None,
        rslora: Optional[bool] = None,
        regression_head: Optional[bool] = None,
        device: str = None
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, processor = FastVisionModel.from_pretrained(
        model_name = model_name,
        load_in_4bit = quantize,
        use_gradient_checkpointing = True,
        max_seq_length = max_seq_len,
        dtype = precision_type
    )
    
    if lora:
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers = lora_layers['vision'],
            finetune_language_layers = lora_layers['language'],
            finetune_attention_modules = lora_layers['attention'],
            finetune_mlp_modules = lora_layers['mlp'],
            r = lora_r,
            lora_alpha = lora_alpha,
            lora_dropout =  lora_dropout,
            bias = "none",
            random_state = 3697,
            use_rslora = rslora
        )
        print("------- LoRA Trainable parameters -------")
        model.print_trainable_parameters()
    
    if regression_head:
        pass

    return model, processor

class RegressionHead(nn.module):
    def __init__(self, hidden_state_dim: int):
        super().__init__()
        self.hidden_state_dim = hidden_state_dim
        
    def forward(self, x):
        return