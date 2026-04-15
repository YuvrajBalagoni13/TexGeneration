import os

dataset_path = "CurrentDataset/txt"

for path in os.listdir(dataset_path):
    type_path = os.path.join(dataset_path, path)
    count = 0
    for folder in os.listdir(type_path):
        count += len(os.listdir(os.path.join(type_path, folder)))
    print(f"{path} - {count}")