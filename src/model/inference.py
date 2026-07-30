# from unsloth import FastVisionModel
import torch
from peft import PeftModel
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
from PIL import Image
from pathlib import Path
import json

from typing import Optional, Any

from .model import RegressionHead, TexGenModel
from .utils import load_checkpoint

class Inference:
    def __init__(
            self,
            model_base: str = None,
            ckpt_path: str = None,
            quantize: bool = True,
            max_seq_length : int = 450,
            device: str = None,
            precision_type: Optional[Any] = None,
            new_tokens: bool = False,
            regression_model: bool = True
    ) -> None:
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if precision_type:
            self.precision_type = precision_type
        else:
            self.precision_type = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        self.max_seq_length = max_seq_length

        self.message = [
            {
                "role" : "user",
                "content" : [
                    {"type" : "image"},
                    {"type" : "text", "text" : "Generate a text based shader graph based on the given input image"}
                ]
            }
        ]

        if regression_model:
            vlm_model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_base,
                torch_dtype = self.precision_type,
                device_map = self.device
            )
            regression_head = RegressionHead(
              embed_dim = vlm_model.config.text_config.hidden_size,
              hidden_dim = 512,
              dropout = 0.01
            ).to(self.precision_type).to(self.device)

            self.model = TexGenModel(
                model = vlm_model,
                regression_head = regression_head,
                trigger_token_id = 248077
            )
        else:
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_base,
                torch_dtype = self.precision_type,
                device_map = self.device
            )
                
        self.processor = AutoProcessor.from_pretrained(ckpt_path)
        

        if ckpt_path:
            self.model, self.processor, _, _, _, _ = load_checkpoint(
                base_model = self.model, 
                processor = self.processor, 
                checkpoint_directory = ckpt_path,  
                regression_model = regression_model
            )
        
        self.processor.tokenizer.padding_side = "left"

        if new_tokens:
            new_embeddings = torch.load(Path(ckpt_path) / "new_embeddings.pth", map_location=self.device)

            if isinstance(new_embeddings, dict):
                new_embeddings = new_embeddings[list(new_embeddings.keys())[0]]

            self.model.model.resize_token_embeddings(len(self.processor.tokenizer), pad_to_multiple_of=None)

            n = new_embeddings.shape[0]
            self.model.model.get_input_embeddings().weight.data[-n:] = new_embeddings.to(self.precision_type)
        
        in_emb = self.model.model.get_input_embeddings()
        out_emb = self.model.model.get_output_embeddings()
        
        print("tied:", in_emb.weight.data_ptr() == out_emb.weight.data_ptr())
        print(self.model.model.active_adapters())
        print(self.model.model.peft_config)

        vocab_size = len(self.processor.tokenizer)
        last_n_ids = list(range(vocab_size - n, vocab_size))
        print(self.processor.tokenizer.convert_ids_to_tokens(last_n_ids))
        
        print("final embedding size:", self.model.model.get_input_embeddings().weight.shape[0])
        print("tokenizer size:", len(self.processor.tokenizer))
        self.model.model.eval()
    
    def infer(
            self,
            image_path : str = None,
            prompt : str = None
    ) -> str:
        image = Image.open(image_path)
        text = self.processor.apply_chat_template(
            self.message,
            tokenize = False,
            add_generation_prompt = True
        )
        inputs = self.processor(
            text = [text],
            images = [image],
            return_tensors = "pt",
            padding = True
        ).to(self.device)

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.precision_type)

        with torch.no_grad():
            output, num_preds = self.model.generate(
                inputs,
                max_new_tokens = self.max_seq_length,
                do_sample = True,
                temperature = 0.3,
                top_p = 0.95
            )
        
        input_length = inputs['input_ids'].shape[1]
        return self.processor.decode(output[0][input_length:]), num_preds.cpu().numpy().tolist()
    
    def batch_infer(
            self,
            image_paths : list[str] = None,
            prompt : str = None
    ) -> dict:
        images = [
            Image.open(image) for image in image_paths
        ]
        texts = [
            self.processor.apply_chat_template(
                self.message,
                tokenize = False,
                add_generation_prompt = True
            ) for _ in image_paths
        ]
        print(repr(texts))
        inputs = self.processor(
            text = texts,
            images = images,
            return_tensors = "pt",
        ).to(self.device)

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.precision_type)

        with torch.no_grad():
            outputs_list, regression_preds_list = self.model.generate(
                inputs,
                max_new_tokens = self.max_seq_length,
                do_sample = False,
                temperature = 0.3,
                top_p = 0.95,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )
            
        print("eos:", self.processor.tokenizer.eos_token_id)
        print("pad:", self.processor.tokenizer.pad_token_id)
        print("last prompt token:", inputs['input_ids'][0, -1].item())
        print("prompt length:", inputs['input_ids'].shape[1])
        
        results = {}
        for i, new_tokens in enumerate(outputs_list):
            decoded = self.processor.decode(new_tokens, skip_special_tokens=True)
            preds = regression_preds_list[i]
            results[str(image_paths[i])] = {
                "shader_text": decoded,
                "nums": preds.cpu().float().numpy().tolist() if preds is not None else []
            }
        return results
        
class SecondInference:
    def __init__(
            self,
            model_base: str = None,
            ckpt_path: str = None,
            quantize: bool = True,
            max_seq_length : int = 450,
            device: str = None,
            precision_type: Optional[Any] = None,
            new_tokens: bool = False,
            regression_model: bool = True
    ) -> None:
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if precision_type:
            self.precision_type = precision_type
        else:
            self.precision_type = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        self.max_seq_length = max_seq_length

        self.message = [
            {
                "role" : "user",
                "content" : [
                    {"type" : "image"},
                    {"type" : "text", "text" : "Generate a text based shader graph based on the given input image"}
                ]
            }
        ]

        if regression_model:
            vlm_model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_base,
                torch_dtype = self.precision_type,
                device_map = self.device
            )
            regression_head = RegressionHead()
            self.model = TexGenModel(
                model = vlm_model,
                regression_head = regression_head,
                trigger_token_id = 248077
            )
        else:
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_base,
                torch_dtype = self.precision_type,
                device_map = self.device
            )
                
        self.processor = AutoProcessor.from_pretrained(lora_path)

        if ckpt_path:
            self.model, self.processor = load_checkpoint(
                base_model = self.model, 
                processor = self.processor, 
                checkpoint_directory = ckpt_path,  
                regression_model = regression_model
            )

        if new_tokens:
            new_embeddings = torch.load(Path(lora_path) / "new_embeddings.pth", map_location=self.device)

            if isinstance(new_embeddings, dict):
                new_embeddings = new_embeddings[list(new_embeddings.keys())[0]]

            self.model.resize_token_embeddings(len(self.processor.tokenizer), pad_to_multiple_of=None)

            n = new_embeddings.shape[0]
            self.model.get_input_embeddings().weight.data[-n:] = new_embeddings.to(self.precision_type)

        print("final embedding size:", self.model.get_input_embeddings().weight.shape[0])
        print("tokenizer size:", len(self.processor.tokenizer))
        self.model.eval()
    
    def infer(
            self,
            image_path : str = None,
            prompt : str = None
    ) -> str:
        image = Image.open(image_path)

        text = self.processor.apply_chat_template(
            self.message,
            add_generation_prompt = True
        )
        inputs = self.processor(
            text = [text],
            images = [image],
            return_tensors = "pt",
        ).to(self.device)

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.precision_type)

        with torch.no_grad():
            output, num_preds = self.model.generate(
                **inputs,
                max_new_tokens = self.max_seq_length,
                do_sample = True,
                temperature = 0.3,
                top_p = 0.95
            )
        
        input_length = inputs['input_ids'].shape[1]
        return self.processor.decode(output[0][input_length:]), num_preds.cpu().numpy().tolist()
    
    def batch_infer(
            self,
            image_paths : list[str] = None,
            prompt : str = None
    ) -> dict:
        images = [
            Image.open(image) for image in image_paths
        ]
        texts = [
            self.processor.apply_chat_template(
                self.message,
                add_generation_prompt = True
            ) for _ in image_paths
        ]
        inputs = self.processor(
            text = texts,
            images = images,
            return_tensors = "pt",
        ).to(self.device)

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.precision_type)

        with torch.no_grad():
            outputs, num_preds = self.model.generate(
                **inputs,
                max_new_tokens = self.max_seq_length,
                do_sample = True,
                temperature = 0.3,
                top_p = 0.95,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )
        
        results = {}
        for i, output in enumerate(outputs):
            input_length = inputs['input_ids'][i].shape[0]
            decoded = self.processor.decode(output[input_length:], skip_special_tokens=True)
            results[str(image_paths[i])] = {
                "shader_text": decoded,
                "nums": num_preds.cpu().numpy().tolist()
            }
        return results
    
if __name__ == "__main__":
    # print("working!")
    inference = Inference(
        model_base = "Unsloth/Qwen3.5-0.8B",
        lora_path = "artifacts/texgen_lora_LoRA_token_main:v12/texgen_LoRA_token_main_1_1750",
        quantize = True,
        max_seq_length = 450,
        device = 'cpu',
        new_tokens = True
    )
    output = inference.infer(image_path="ShaderDataset/val/mat_llm/case_00000_gen_02/00001.jpg")
    print(output)