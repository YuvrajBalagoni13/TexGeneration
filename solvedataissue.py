from pathlib import Path
import shutil
from tqdm.auto import tqdm

data_path = Path("CurrentDataset/txt")

for txt in tqdm(data_path.iterdir()):
    txt_name = str(txt.name) 
    splitted_txt_name = txt_name.split("-")
    current_path = data_path
    for directory in splitted_txt_name[:-1]:
        current_path = current_path / directory
        current_path.mkdir(parents=True, exist_ok=True)
    shutil.move(txt, current_path / splitted_txt_name[-1])

