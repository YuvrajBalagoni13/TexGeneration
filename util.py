import os
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path
import shutil

# dataset_path = Path("material_dataset_filtered")
# txt_data_save_path = Path("Datasets")

# py_files = list(dataset_path.rglob("*.py"))

# for file in py_files:
#     txt_file_name = "-".join(str(file.with_suffix(".txt")).split("/")[1:])
#     txt_file_path = txt_data_save_path / txt_file_name
#     print(txt_file_path)
#     print(txt_file_name)
#     break

data_path = Path("CurrentDataset/txt")

for txt_file in tqdm(data_path.iterdir()):
    with open(txt_file, "r") as f:
        txt = f.read().lower()
        if "rgbcurve" in txt:
            print(txt_file)
            print(txt)
            break
