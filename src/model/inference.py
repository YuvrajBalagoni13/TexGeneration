from unsloth import FastVisionModel
import torch
from transformers import TextStreamer
from PIL import Image

class Inference:
    def __init__(self, model_name: str = "unsloth/Qwen3.5-2B", load_in_4bit: bool = True, device: str = None) -> None:
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit = load_in_4bit,
            use_gradient_checkpointing = "unsloth"
        )
        FastVisionModel.for_inference(self.model)

    def infer(self, image_path: str = "", input_prompt: str = "") -> str:
        messages = [
            {"role" : "user",
             "content" : [
                 {"type": "image"},
                 {"type": "text", "text" : input_prompt}
             ]}
        ]
        input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt = True)
        inputs = self.tokenizer(
            Image.open(image_path),
            input_text,
            add_special_tokens = False,
            return_tensors = "pt",
        ).to(self.device)

        text_streamer = TextStreamer(self.tokenizer, skip_prompt = True)
        output = self.model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                   use_cache = True, temperature = 1.5, min_p = 0.1)
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return decoded
    

if __name__ == "__main__":
    inference = Inference()
    inference.infer(
        image_path="Dataset/infinigen/bone/00000.jpg",
        input_prompt="generate a blender procedural shader graph in a text based shader which is structured in the following way-\nN|node_variable_name:node_type_name;(rest of the nodes)\nP|node_property_path:value;(rest of the properties)\nL|output_node.out_socket>input_node.in_socket;(rest of the links)\nDon't write anything else only these 3 lines in the structure mentioned"
    )
