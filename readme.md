# TexGen : Inverse Procedural Material Modelling via VLM.

TexGen is a inverse procedural material model that takes a texture image as input & generates a Blender procedural material graphs for that specific image for highly customizable, editable material workflows.

# Overview 
Existing work on material generation focus on synthesizing image based texture maps which are baked & static. TexGen tries to generate the procedural shader graph for the given image which are editable, giving more control in the overall workflow.

# Usage 
### Using Addon
install the TexAddon.zip file from the repo & install it in blender by going -
```
Edit -> Preference -> install from disk -> TexAddon.zip
```
now the addon will be visible in the properties -> material -> TexGen
before using the addon it requires an python environment for it.

Create conda environment for the addon -
```
conda create -n TexGen python=3.13 # blender 5.1 uses 3.13 python 
conda activate TexGen
pip install -r requirements.txt
python -c "import site; print(site.getsitepackages()[0])" # this tells where the env packages are add this to env_dir in the addon panel
```
Now add this en




