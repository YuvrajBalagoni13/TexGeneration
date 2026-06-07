from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict
import json
import random

import sys
import os
sys.path.insert(0, '/home/ML/TextureGeneration')

from src.data.txt_shader import TextShader

def replace_in_file(file_path: str, old: str, new: str) -> None:
    with open(file_path, "r") as f:
        content = f.read()
    with open(file_path, "w") as f:
        f.write(content.replace(old, new))

dataset_path = Path("ShaderDataset")
errors = {}
errors['error_samples'] = []
errors['file_issue'] = []

text_shader_validator = TextShader()

images_list = list(dataset_path.rglob("*.jpg"))
print(len(images_list))
total = 0
for image in tqdm(images_list):
    text_path = image.with_suffix('.txt')
    if not image.exists() or not text_path.exists():
        print("doesn't exists :(")
        errors['file_issue'].append(str(image))
    try:
        mat = text_shader_validator.text_to_shader_graph(text_shader_path=text_path)
    except Exception as e:
        errors['error_samples'].append(str(image))
        total += 1
    finally:
        text_shader_validator.cleanup_material()
    
print(f"----- Total samples with errors {total} / {len(images_list)} -----")
with open("errors.json", "w") as f:
    json.dump(errors, f, indent=4)