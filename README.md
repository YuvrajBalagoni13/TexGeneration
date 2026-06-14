# TexGen : Inverse Procedural Material Modelling via VLM.

TexGen is a inverse procedural material model that takes a texture image as input & generates a Blender procedural material graphs for that specific image for highly customizable, editable material workflows.

# Overview 
Existing work on material generation focus on synthesizing image based texture maps which are baked & static. TexGen tries to generate the procedural shader graph for the given image which are editable, giving more control in the overall workflow.

```
Input Image -> fine-tuned VLM -> DSL based shader -> addon creates the shader graph
```

# Results / Examples


# Key Highlight of project -
- **Inverse Procedural Modelling** : Given an image the model will generate a shader graph corresponding to that image.
- **Dataset** : Converted VLMaterial dataset (target was a python script for the material) into a Domain specific language for shader graph. Why & How in this doc [DATA.md](docs/DATA.md).
- **Custom tokens** : added new additional domain specific tokens to the models tokenizer vocabulary for efficient shader representation. In details at [DATA.md](docs/DATA.md).
- **Model** : did lora fine-tuning for Qwen3.5-0.8B model, also trained the additional embeddings for the new added tokens. Also untied the weights for the new embeddings & lm head as untying helped model performed & learned better than keeping them same. More info here [TRAIN.md](docs/TRAIN.md).
- **Quantization** : Quantized the model into multiple precision (Q8_0, Q5_K_M, Q4_K_M) using llama.cpp in gguf format. Available at huggingface - [huggingface_link](https://huggingface.co/YuvrajB13/Qwen-3.5-0.8B-TexGen-GGUF/tree/main).
- **Blender Addon** : Created a blender addon to use the model.

# Usage 
### Installing Addon
install the TexAddon.zip file from the repo & install it in blender by going -
```
Edit -> Preference -> install from disk -> TexAddon.zip
```
now the addon will be visible in the properties -> material -> TexGen
before using the addon it requires an python environment for it.

Create conda environment for the addon -
```bash
conda env create -f addon_environment.yml                   # download from addon folder
conda activate TexGenAddon
python -c "import site; print(site.getsitepackages()[0])"   # this tells where the env packages are add this to env_dir in the addon panel
```
Now add this env path to the Addons preference panel site packages path -
![alt text](docs/imgs/addon_preference.png)

### Using Addon
So the addon will be present in the properties -> material panel
![alt text](docs/imgs/usage_addon.png)

add model path if installed the model manually from huggingface or else the addon will install it for you & save it in huggingface cache.

# Metrics

|Model Precision|LPIPS|CLIP|$e^{-lpips}$|
|---|---|---|---|
|F16|   |   |   |
|Q8_0|   |   |   |
|Q5_K_M|   |   |   |
|Q4_K_M|   |   |   |

Here all the metrics are scaled by $(1 - error\%)$.


# Project Structure
```
TexGeneration/
├── src/
│   ├── model/
│   │   ├── train.py                # training script
│   │   ├── dataset.py              # ShaderDataset
│   │   ├── inference.py            # inference pipeline using transformers
│   │   ├── infer.py   
|   |   ├── embeddings.py           # Custom new embeddings implementation
|   |   ├── merge_lora.py          
|   |   ├── renderer.py             # Render generated samples during eval         
|   |   ├── metric.py               # Calculate metrics
|   |   └── utils.py                # logging, saving & loading ckpts
│   └── data/
|       ├── additional_tokens.py    # Getting tokens not present in tokenizer
|       ├── convert_dataset.py      # converts python data into dsl
|       ├── dsl.py                  # shader code for conversion
│       └── txt_shader.py           # main shader code to create material
├── docs/                           # Yet to come ....
├── addon/                          # Addon folder with essential scripts
│   ├── __init__.py       
│   ├── gguf_inference.py
|   ├── txt_shader.py
|   └── addon_environment.yml
├── TexGen_Addon.zip                # Zip file of addon
├── environment.yml               
└── README.md
```

# Limitations

# Future Work & Ideas