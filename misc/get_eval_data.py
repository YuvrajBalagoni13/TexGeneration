from pathlib import Path
import random
import shutil

no_of_samples = 10
train_dataset = Path("Dataset/train")
test_dataset = Path("Dataset/infinigen")

train_eval_path = Path("Dataset/eval/train_eval")
test_eval_path = Path("Dataset/eval/test_eval")

train_image_list = list(train_dataset.rglob("*.jpg"))
test_image_list = list(test_dataset.rglob("*.jpg"))

train_eval_samples = random.sample(train_image_list, k=no_of_samples)
test_eval_samples = random.sample(test_image_list, k=no_of_samples)

for image_path in train_eval_samples:
    shutil.copy2(image_path, train_eval_path / image_path.name)
    shutil.copy2(image_path.with_suffix(".txt"), train_eval_path / image_path.with_suffix(".txt").name)

for image_path in test_eval_samples:
    shutil.copy2(image_path, test_eval_path / image_path.name)
    shutil.copy2(image_path.with_suffix(".txt"), test_eval_path / image_path.with_suffix(".txt").name)
