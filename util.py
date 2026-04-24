from pathlib import Path
from tqdm.auto import tqdm

dataset_path = Path("Dataset/infinigen")
img_files = list(dataset_path.rglob("*.jpg"))

for file in tqdm(img_files):
    file_name = file.name.split(".")[0][4:-7]
    new_file_path = file.with_name(f"{file_name}.jpg")
    file.rename(new_file_path)

    txt_file = file.with_name(f"var_{file_name}_full.txt")
    txt_file_name = txt_file.name.split(".")[0][4:-5]
    new_txt_file_path = file.with_name(f"{txt_file_name}.txt")
    txt_file.rename(new_txt_file_path)
