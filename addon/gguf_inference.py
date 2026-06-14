import sys

# sys.path.append("/home/anaconda3/envs/TexGen/lib/python3.14/site-packages")

from llama_cpp.llama_chat_format import Qwen35ChatHandler
from llama_cpp import Llama
from PIL import Image
from time import time
from pathlib import Path

class PatchedQwen35ChatHandler(Qwen35ChatHandler):
    PATCHED_TEMPLATE = Qwen35ChatHandler.CHAT_FORMAT \
    .replace("{{- raise_exception('System message cannot contain images.') -}}", "") \
    .replace("{{- raise_exception('llama.cpp does not currently support video.') -}}", "") \
    .replace("{{- raise_exception('System message cannot contain videos.') -}}", "") \
    .replace("{{- raise_exception('Unexpected item type in content.') -}}", "") \
    .replace("{{- raise_exception('Unexpected content type.') -}}", "") \
    .replace("{{- raise_exception('No messages provided.') -}}", "") \
    .replace("{{- raise_exception('No user query found in messages.') -}}", "") \
    .replace("{{- raise_exception('System message must be at the beginning.') -}}", "") \
    .replace("{{- raise_exception('Unexpected message role.') -}}", "")
    CHAT_FORMAT = PATCHED_TEMPLATE 
    DEFAULT_SYSTEM_MESSAGE = ""

class GGUFInference:
    def __init__(
            self,
            model_repo : str = None,
            mmproj_file : str = None,
            model_file: str = None,
            model_path: str | Path = None,
            mmproj_path: str | Path = None,
            n_ctx: int = 800,
            max_tokens: int = 450,
            temperature: float = 0.3,
            top_p: float = 0.95,
            verbose: bool = False
    ) -> None:
        
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        if mmproj_path:
            self.chat_handler = PatchedQwen35ChatHandler(
                clip_model_path = mmproj_path,
                enable_thinking = False,
                verbose = verbose
            )
        else:
            try:
                print("----- Downloading mmproj checkpoints -----")
                self.chat_handler = PatchedQwen35ChatHandler.from_pretrained(
                    repo_id = model_repo,
                    filename = mmproj_file,
                    verbose = verbose
                )
            except Exception as e:
                raise ValueError(e)
        
        if model_path:
            self.vlm = Llama(
                model_path = model_path,
                chat_handler = self.chat_handler,
                n_gpu_layers = -1,
                n_ctx = n_ctx,
                swa_full = True,
                verbose = verbose
            )
        else:
            try:
                print("----- Downloading model checkpoints -----")
                self.vlm = Llama.from_pretrained(
                    repo_id = model_repo,
                    filename = model_file,
                    chat_handler = self.chat_handler,
                    n_gpu_layers = -1,
                    n_ctx = n_ctx,
                    swa_full = True,
                    verbose = verbose
                )
            except Exception as e:
                raise ValueError(e)

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
    
    def infer(
            self,
            image_path: str | Path = None,
            image: Image.Image = None
    ) -> str:
        if not image:
            image = Image.open(image_path)
            image = image.resize((512, 512))
        
        self.message[0]["content"][0]["image"] = image

        time1 = time()
        response = self.vlm.create_chat_completion(
            messages = self.message,
            max_tokens = self.max_tokens,
            temperature = self.temperature,
            top_p = self.top_p
        )
        time2 = time()
        print(f"Inference time = {time2 - time1:.2f} seconds")

        return response["choices"][0]["message"]["content"]
    
if __name__ == "__main__":
    gguf_inference = GGUFInference(
        mmproj_path="mmproj_Qwen3_5_0_8B_UT_F16.gguf",
        model_path="Qwen3_5_0_8B_UT_6750_fixed.gguf",
        n_ctx=1024,
        max_tokens=512,
        temperature=0.3,
        top_p=0.95,
        verbose=False
    )
    output = gguf_inference.infer(image_path="ShaderDataset/train/mat_llm_r4/case_00000_gen_01/00000.jpg")
    print(output)