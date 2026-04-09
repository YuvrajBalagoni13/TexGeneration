import os
import json
from pathlib import Path
from tqdm.auto import tqdm

DATA_PATH = Path("material_dataset_filtered")

total_samples = 0
groups_file_paths = []

for source in tqdm(DATA_PATH.iterdir()):
    if source.is_dir():
        for types in source.iterdir():
            if types.is_dir():
                for files in types.iterdir():
                    if files.suffix == ".txt":
                        with open(files, "r") as f:
                            content = f.read()
                            if "group" in content:
                                total_samples += 1
                                groups_file_paths.append(files)

serializable_paths = [str(path.with_suffix(".py")) for path in groups_file_paths]

print(total_samples)
with open("groups.json", "w") as f:
    json.dump({"groups": serializable_paths}, f, indent=4)