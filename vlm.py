from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image
from transformers.image_utils import load_image
from rag_code.retrieve import SampleRetriever
from time import time

class ModelInference:
    def __init__(
        self,
        model_path : str = "HuggingFaceTB/SmolVLM-500M-Instruct",
        num_retrieval: int = 2,
        device: str = "cpu"
    ) -> None:

        self.num_retrieval = num_retrieval

        if device is not None:  
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print("Loading model ...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(model_path,
                                        torch_dtype=torch.float16,
                                        _attn_implementation="flash_attention_2" if device == "cuda" else "eager").to(self.device)

        print("Loading Retriever ...")
        self.retriever = SampleRetriever(
            num_results = num_retrieval,
            device = self.device
        )

        self.system_message = {
                "role": "system",
                "content": [
                   {"type": "text", "text": "You are an expert in texture analysis and node-based shader programming. Your task is to analyze an input image and generate a valid Node JSON representation that describes its material properties (e.g., base color, roughness, metallic, normal mapping)."}
                ]
            }
    
    def get_node_json(
        self,
        input_image_path: str = None
    ) -> dict:

        print("Retrieving items ...")
        try:
            retrieved_items = self.retriever.retrieve(input_image_path)
        except SpecificException as e:
            print(f"RetrievingError: {e}")  

        retrieved_image_jsons = retrieved_items['documents'][0]
        retrieved_image_metadata = retrieved_items['metadatas'][0]

        current_message = [self.system_message]
        images = []
        for i in range(self.num_retrieval):
            current_message.append({
                "role" : "user",
                "content" : [
                    {"type" : "text", "text" : f"Example {i+1} - Reference Image: "},
                    {"type" : "image"}, # Retrieved images
                    {"type" : "text", "text" : f"Resulting Node JSON:\n{retrieved_image_jsons[i]}"}
                ]
            })
            images.append(load_image(retrieved_image_metadata[i]['image_path']))
        images.append(load_image(input_image_path))

        current_message.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Now, analyze this new Input Image and generate its corresponding Node JSON. Ensure the JSON follows the exact schema shown in the examples above. Output ONLY the JSON code block."},
                {"type": "image"} # Input image
            ]
        })

        print("getting output ...")
        start_time = time()
        prompt = self.processor.apply_chat_template(current_message, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(self.device)

        input_len = inputs.input_ids.shape[1]

        generated_ids = self.model.generate(
            **inputs, 
            max_new_tokens=500,
            do_sample=False,       # Greedy search is better for structured JSON
            repetition_penalty=1.2 # Stops it from repeating "pink squares..."
            )
        generated_texts = self.processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
        )
        end_time = time()
        print(f"Done -> took {end_time - start_time} seconds.")
        return generated_texts[0]


if __name__ == "__main__":
    inference = ModelInference()
    node = inference.get_node_json("bsdf_bricktex_dataset/test/images/ShaderNodeBsdfPrincipled_ShaderNodeTexBrick_000_095.png")
    print(node)