from unsloth import FastVisionModel
import torch
from peft import PeftModel
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
from PIL import Image
from pathlib import Path
import json

class UnslothInference:
    def __init__(
            self, 
            model_base: str = None, 
            lora_path : str = None,
            quantize: bool = True, 
            device: str = None,
            max_seq_length : int = 768,
            new_tokens : bool = True
            ) -> None:
        
        self.max_seq_length = max_seq_length

        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.precision_type = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        self.message = [
            {"role" : "user",
             "content" : [
                 {"type": "image"},
                 {"type": "text", "text" : (
                     "Generate a text based shader graph in the following format -\n"
                     "N|node_name:node_type;...\n"
                     "P|node_name.property_path:value;...\n"
                     "L|node_name.output_socket>node_name.input_socket;...\n"
                     "Here N| represents nodes, P| tells properties & L| tells links."
                 )}
             ]}
        ]

        self.model, self.processor = FastVisionModel.from_pretrained(
            model_base,
            load_in_4bit = quantize,
            use_gradient_checkpointing = False,
            max_seq_length = self.max_seq_length,
            dtype = self.precision_type
        )
        self.model = PeftModel.from_pretrained(
            self.model,
            lora_path
        )

        if new_tokens:
            new_embeddings = torch.load(Path(lora_path) / "new_embeddings.pth", map_location=self.device)
            new_lm_head = torch.load(Path(lora_path) / "new_lm_head.pth", map_location=self.device)

            if isinstance(new_embeddings, dict):
                new_embeddings = new_embeddings[list(new_embeddings.keys())[0]]
            if isinstance(new_lm_head, dict):
                new_lm_head = new_lm_head[list(new_lm_head.keys())[0]]

            self.model.resize_token_embeddings(len(self.processor.tokenizer), pad_to_multiple_of=None)

            n = new_embeddings.shape[0]
            self.model.get_input_embeddings().weight.data[-n:]  = new_embeddings.to(self.precision_type)
            self.model.get_output_embeddings().weight.data[-n:] = new_lm_head.to(self.precision_type)

        print("final embedding size:", self.model.get_input_embeddings().weight.shape[0])
        print("tokenizer size:", len(self.processor.tokenizer))

        FastVisionModel.for_inference(self.model)

    def infer(
            self, 
            image_path: str = "", 
            input_prompt: str = ""
            ) -> str:
        
        input_text = self.processor.apply_chat_template(self.message, add_generation_prompt = True)
        inputs = self.processor(
            Image.open(image_path),
            input_text,
            add_special_tokens = False,
            return_tensors = "pt",
        ).to(self.device)

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.precision_type)

        output = self.model.generate(**inputs, max_new_tokens = self.max_seq_length,
                   use_cache = True, do_sample = True, temperature = 0.5, min_p = 0.1)
        input_length = inputs["input_ids"].shape[1]
        decoded = self.processor.decode(output[0][input_length:], skip_special_tokens=True)
        return decoded
    
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
            max_length = self.max_seq_length
        ).to(self.device)

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.precision_type)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens = self.max_seq_length,
                do_sample = True,
                temperature = 0.5,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
        
        results = {}
        for i, output in enumerate(outputs):
            input_length = inputs['input_ids'][i].shape[0]
            decoded = self.processor.decode(output[input_length:])
            results[str(image_paths[i])] = decoded
        return results
    
class Inference:
    def __init__(
            self,
            model_base: str = None,
            lora_path: str = None,
            quantize: bool = True,
            max_seq_length : int = 450,
            device: str = None,
            precision_type: any = None,
            new_tokens: bool = False
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
                    {"type" : "text", "text" : (
                     "Generate a text based shader graph in the following format -\n"
                     "N|node_name:node_type;...\n"
                     "P|node_name.property_path:value;...\n"
                     "L|node_name.output_socket>node_name.input_socket;...\n"
                     "Here N| represents nodes, P| tells properties & L| tells links."
                    )}
                ]
            }
        ]

        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_base,
            torch_dtype = self.precision_type,
            device_map = self.device
        )
        self.processor = AutoProcessor.from_pretrained(lora_path)

        if lora_path:
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_path
            )

        if new_tokens:
            new_embeddings = torch.load(Path(lora_path) / "new_embeddings.pth", map_location=self.device)
            new_lm_head = torch.load(Path(lora_path) / "new_lm_head.pth", map_location=self.device)

            if isinstance(new_embeddings, dict):
                new_embeddings = new_embeddings[list(new_embeddings.keys())[0]]
            if isinstance(new_lm_head, dict):
                new_lm_head = new_lm_head[list(new_lm_head.keys())[0]]

            self.model.resize_token_embeddings(len(self.processor.tokenizer), pad_to_multiple_of=None)

            n = new_embeddings.shape[0]
            self.model.get_input_embeddings().weight.data[-n:]  = new_embeddings.to(self.precision_type)
            self.model.get_output_embeddings().weight.data[-n:] = new_lm_head.to(self.precision_type)

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
            max_length = self.max_seq_length
        ).to(self.device)

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.precision_type)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens = self.max_seq_length,
                do_sample = True,
                temperature = 0.5,
                top_p = 0.95
            )
        
        input_length = inputs['input_ids'].shape[1]
        return self.processor.decode(output[0][input_length:])
    
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
            # max_length = self.max_seq_length
        ).to(self.device)

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.precision_type)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens = self.max_seq_length,
                do_sample = False,
                temperature = 0.0,
                repition_penalty = 1.1,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
            )
        
        results = {}
        for i, output in enumerate(outputs):
            input_length = inputs['input_ids'][i].shape[0]
            decoded = self.processor.decode(output[input_length:], skip_special_tokens=True)
            results[str(image_paths[i])] = decoded
        return results
    
if __name__ == "__main__":
    print("working!")
    # inference = Inference()
    # inference.infer(
    #     image_path="Dataset/infinigen/bone/00000.jpg",
    #     input_prompt="generate a blender procedural shader graph in a text based shader which is structured in the following way-\nN|node_variable_name:node_type_name;(rest of the nodes)\nP|node_property_path:value;(rest of the properties)\nL|output_node.out_socket>input_node.in_socket;(rest of the links)\nDon't write anything else only these 3 lines in the structure mentioned"
    # )
