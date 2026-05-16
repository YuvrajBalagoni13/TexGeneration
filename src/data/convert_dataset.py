import sys
import os

# Get the directory where convert_dataset.py is located
script_dir = os.path.dirname(os.path.realpath(__file__))

# Add that directory to the search path
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Now you can import dsl
import dsl
import bpy

from pathlib import Path
import json
from tqdm.auto import tqdm 
import shutil


def get_path_from_name(name, data_path):
    splitted_name = name.split("-")
    current_path = data_path
    for name in splitted_name[:-1]:
        current_path = current_path / name 
        current_path.mkdir(parents=True, exist_ok=True)
    current_path = current_path / splitted_name[-1]
    return current_path

convert_code = dsl.ConvertCodeToDSL()

dataset_path = Path("material_dataset_filtered/mat_llm_r3")
data_save_folder = Path("ShaderDataset")
print("Starting conversion ...")

with open("JSON_files/groups.json", "r") as f:
    groups_dict = json.load(f)

group_set = set()
for grp in groups_dict["groups"]:
    if grp in group_set: continue
    else: group_set.add(grp)

py_files = list(dataset_path.rglob("*.py"))

count = 0
for file_path in tqdm(py_files):
    if str(file_path) in group_set:
        continue

    txt_file_name = "-".join(str(file_path.with_suffix(".txt")).split("/")[1:])
    txt_file_path = get_path_from_name(txt_file_name, data_save_folder)

    file_path_parts = str(file_path).split("_")
    file_path_parts[-1] = "render.jpg"
    img_source_path = "_".join(file_path_parts)

    save_file_path_parts = str(txt_file_path).split("_")
    save_file_path_parts[-1] = "render.jpg"
    img_save_path = "_".join(save_file_path_parts)

    try:
        text = convert_code.convert(file_path, txt_file_path)
        shutil.copy(img_source_path, img_save_path)

        count += 1
    except Exception as e:
        print(f"{file_path} : {e}")
        continue
    # break

print("completed conversions...")
print(f"converted {count} .py files into .txt successfully")

"""
/mnt/Storage/ML/blender-5.1.0-linux-x64/blender --background --python src/data/convert_dataset.py
"""