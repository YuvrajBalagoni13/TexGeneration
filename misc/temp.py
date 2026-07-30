from pathlib import Path
from tqdm import tqdm

dataset_path = Path("ShaderDataset")

data_list = list(dataset_path.iterdir())

res = {}

for data_path in data_list:
    unique_images = list(data_path.rglob("*.jpg"))
    cnt = 0
    print(str(data_path))
    for images in tqdm(unique_images):
        if images.with_suffix(".txt").is_file():
            cnt += 1
    res[str(data_path)] = cnt

print(res)
