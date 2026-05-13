#!/bin/bash
# evaluate.sh

echo "--- Step 1: Running inference ---"
python -m src.model.inference_pass

echo "--- Step 2: Rendering in Blender ---"
/mnt/Storage/ML/blender-5.1.0-linux-x64/blender --background --python src/model/blender_render.py

echo "--- Step 3: Computing LPIPS scores ---"
python -m src.model.scoring

echo "--- Done ---"