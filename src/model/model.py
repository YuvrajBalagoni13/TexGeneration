import torch
import torch.nn as nn

from unsloth import FastVisionModel
from typing import Optional, Dict, Any

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
    def __init__(self, model, regression_head, trigger_token_id=248077):
        super().__init__()
        self.model = model
        self.regression_head = regression_head
        self.trigger_token_id = trigger_token_id

        last_layer = self.model.base_model.language_model.layers[-1]
        last_layer.register_forward_hook(self._capture_hidden)

    def _capture_hidden(self, module, input, output):
        self._last_hidden_state = output[0] if isinstance(output, tuple) else output

    @staticmethod
    def get_last_layer(model_instance):
        current = model_instance
        while True:
          if hasattr(current, 'base_model') and current.base_model is not current:
            current = current.base_model
            print("base_model")
          elif hasattr(current, 'model') and current.model is not current:
            current = current.model
            print("model")
          elif hasattr(current, 'language_model') and current.language_model is not current:
            current = current.language_model
            print("language_model")
          else:
            break
        
        if hasattr(current, 'layers'):
          return current.layers[-1]
        raise AttributeError(f"Couldn't find transformer layers in {type(current).__name__}")

    def forward(self, x):
        model_inputs = {k: v for k, v in x.items() if k != "labels"}

        outputs = self.model(**model_inputs)
        vlm_logits = outputs.logits

        last_hidden_states = self._last_hidden_state
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
            max_new_tokens: int = 512,
            do_sample: bool = False,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            pad_token_id: Optional[Any] = None,
            eos_token_id: Optional[Any] = None
    ):
        
        outputs = self.model.generate(
            **x,
            max_new_tokens = max_new_tokens,
            output_hidden_states = True,
            return_dict_in_generate = True,
            do_sample = do_sample,
            temperature = temperature,
            top_p = top_p,
            pad_token_id = pad_token_id,
            eos_token_id = eos_token_id
        )

        prompt_length = x['input_ids'].shape[1]
        batch_size = outputs.sequences.shape[0]

        all_new_tokens = []
        all_regression_preds = []

        for batch_idx in range(batch_size):
            full_sequence = outputs.sequences[batch_idx]
            new_tokens = full_sequence[prompt_length:]
            all_new_tokens.append(new_tokens)

            valid_hidden_states = []
            for step_idx, token_id in enumerate(new_tokens):
                if token_id == self.trigger_token_id:
                    # outputs.hidden_states[step_idx][-1] shape: (batch_size, 1, hidden_dim) during generation
                    step_hidden_state = outputs.hidden_states[step_idx][-1][batch_idx, 0, :]
                    valid_hidden_states.append(step_hidden_state)

            if valid_hidden_states:
                stacked_states = torch.stack(valid_hidden_states)
                regression_preds = self.regression_head(stacked_states).squeeze(-1)
            else:
                regression_preds = None
            all_regression_preds.append(regression_preds)

        return all_new_tokens, all_regression_preds
    
class RegressionHead(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.linear(x)