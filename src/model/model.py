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
    return model, processor

class TexGenModel(nn.Module):
    def __init__(self, model, regression_head, trigger_token_id):
        super().__init__()
        self.model = model
        self.regression_head = regression_head
        self.trigger_token_id = trigger_token_id

    def forward(self, x):
        outputs = self.model(**x, output_hidden_states=True)
        vlm_logits = outputs.logits

        last_hidden_states = outputs.hidden_states[-1] 
        num_mask = (x['input_ids'] == self.trigger_token_id)
        valid_hidden_states = last_hidden_states[num_mask]

        regression_preds = None
        if valid_hidden_states.numel() > 0:
            regression_preds = self.regression_head(valid_hidden_states)
        return vlm_logits, regression_preds
    
    @torch.no_grad()
    def generate(
            self,
            x,
            max_new_tokens: int = 512
    ):
        outputs = self.model.generate(
            **x,
            max_new_tokens = max_new_tokens,
            output_hidden_states = True,
            return_dict_in_generate = True
        )

        prompt_length = x['input_ids'].shape[1]
        full_sequence = outputs.sequences[0]
        new_tokens = full_sequence[prompt_length:]

        valid_hidden_states = []

        for step_idx, token_id in enumerate(new_tokens):
            if token_id == self.trigger_token_id:
                step_hidden_state = outputs.hidden_states[step_idx][-1][0, 0, :]
                valid_hidden_states.append(step_hidden_state)
        
        numerical_preds = None
        if valid_hidden_states:
            stacked_states = torch.stack(valid_hidden_states)
            numerical_preds = self.regression_head(stacked_states)
        
        return new_tokens, numerical_preds
    
class RegressionHead(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.linear(x)