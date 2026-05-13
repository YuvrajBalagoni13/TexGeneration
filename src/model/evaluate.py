import torch
from tqdm.auto import tqdm
import json
import bpy
import os
from pathlib import Path
import lpips
import torchvision.transforms as T
from PIL import Image

from .inference import Inference
from ..data.txt_shader import TextShader

def setup_scene():
    """Clears the scene and sets up the camera and plane."""
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()
    
    # Setup camera
    cam_data = bpy.data.cameras.new('camera')
    cam = bpy.data.objects.new('camera', cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.location = (0.0, 0.0, 2.75)

    # Add plane & make it active
    bpy.ops.mesh.primitive_plane_add()
    return bpy.context.active_object

def setup_generalized_lighting():
    """Sets up a procedural Nishita sky dome for highly realistic, generalized environment lighting."""
    
    # Get the current world scene
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    
    # Clear existing world nodes to start fresh
    nodes.clear()
    
    # Create the necessary world nodes
    world_out = nodes.new(type='ShaderNodeOutputWorld')
    bg_node = nodes.new(type='ShaderNodeBackground')
    sky_node = nodes.new(type='ShaderNodeTexSky')
    
    # Configure the Sky Texture (Nishita is Blender's most realistic sky model)
    sky_node.sky_type = 'HOSEK_WILKIE'
    
    # Tweak the sun so it isn't pointing straight down. 
    # An angle creates gradients on the plane, allowing the ML model to see how light falls off.
    sky_node.sun_elevation = 0.6 # Height of the sun (roughly 35 degrees)
    sky_node.sun_rotation = 2.5   # Rotation around the Z axis
    sky_node.sun_intensity = 0.5  # Slightly lowered so it doesn't blow out bright materials
    
    # Link them together: Sky -> Background -> World Output
    links.new(sky_node.outputs['Color'], bg_node.inputs['Color'])
    links.new(bg_node.outputs['Background'], world_out.inputs['Surface'])
    bpy.ops.object.light_add(type='SUN', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    bpy.context.object.data.energy = 4

def set_render_settings() -> None:
    scene_render = bpy.context.scene.render
    scene_render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 128
    bpy.context.scene.cycles.use_denoising = False
    scene_render.image_settings.file_format = 'JPEG'
    scene_render.resolution_x = 512
    scene_render.resolution_y = 512

def add_material_to_plane(
        plane: bpy.types.Object = None, 
        material: bpy.types.Material = None, 
        image_path: str = None
        ) -> str:
    """return path to rendered image"""
    plane.data.materials.clear() 
    plane.data.materials.append(material)
    
    path = Path(image_path)
    render_image_path = path.parent / f"render_{path.stem}.jpg"

    bpy.context.scene.render.filepath = render_image_path
    bpy.ops.render.render(write_still=True)
    return render_image_path
    
    
def load_eval_data(eval_data_path: str = "") -> list[dict]:
    image_path_list = list(Path(eval_data_path).rglob("*.jpg"))
    data = []
    for image_path in image_path_list:
        sample = {}
        sample["image"] = str(image_path)
        with open(image_path.with_suffix(".txt"), "r") as f:
            sample["shader"] = f.read()
        data.append(sample)
    print(f"---------- Total eval data length = {len(data)} samples ----------")
    return data

def load_image_tensor(path: str = "") -> torch.Tensor:
    try:
        img = Image.open(path).convert("RGB").resize((224, 224))
        tensor = T.ToTensor()(img).unsqueeze(0)
        tensor = tensor * 2 - 1 # normalizing between [-1, 1] for lpips
        return tensor
    except Exception as e:
        raise ValueError(f"Error - {e}")

def similarity_score(
        image_path: str = "",
        render_path: str = "",
        lpips: any = None,
        device: str = None
        ) -> float:
    image_tensor = load_image_tensor(image_path).to(device)
    render_tensor = load_image_tensor(render_path).to(device)

    score = lpips(image_tensor, render_tensor).item()
    return score

def main(
        data_path: str = "",
        save_report_path: str = ""
) -> None:
    # load a group of data to evaluate the model
    eval_data = load_eval_data(eval_data_path = data_path)

    # Get response from the model
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    inference_model = Inference(
        model_name = "model_ckpts/run_0.1_lora",
        load_in_4bit = True,
        device = device
    )

    input_prompt = ("Generate a text based shader graph in the following format -\n"
                    "N|node_name:node_type;...\n"
                    "P|node_name.property_path:value;...\n"
                    "L|node_name.output_socket>node_name.input_socket;...\n"
                    "Here N| represents nodes, P| tells properties & L| tells links.")
    
    txt_shader = TextShader()

    lpips_loss = lpips.LPIPS(net = 'vgg').to(device)

    avg_score = 0.0
    final_report = {}

    plane = setup_scene()
    setup_generalized_lighting()
    set_render_settings()

    for i, sample in tqdm(enumerate(eval_data)):
        log_error = None
        sample_report = {}
        sample_report["image_path"] = str(sample["image"])
        sample_report["actual_shader"] = sample["shader"]

        output_txt_shader = inference_model.infer(
            image_path = sample["image"],
            input_prompt = input_prompt
        )

        sample_report["output_shader"] = output_txt_shader

        render_image_path = None
        try:
            # pass them through blender text shader 
            material = txt_shader.text_to_shader_graph(
                text_shader=output_txt_shader,
                material_name=sample["image"].split("/")[-1].split(".")[0]
            )

            render_image_path = add_material_to_plane(
                plane = plane, 
                material = material, 
                image_path = sample["image"]
                )
        except Exception as e:
            log_error = f"Error: {e}"

        sample_report["error"] = log_error

        # calculate a metric
        if log_error is not None:
            final_score = 1.0
        else:
            final_score = similarity_score(
                image_path = sample["image"],
                render_path = render_image_path,
                lpips = lpips_loss,
                device = device
            )
            
        avg_score = avg_score + final_score
        sample_report["score"] = float(final_score)

        final_report[f"sample_{i}"] = sample_report

    # log & print results
    avg_score = avg_score / len(eval_data)
    final_report["avg_score"] = avg_score

    with open(save_report_path, "w") as f:
        json.dump(final_report, f, indent=4)

if __name__ == "__main__" :
    eval_data_path = "Dataset/eval"
    train_eval_data_path = "Dataset/eval/train_eval"
    test_eval_data_path = "Dataset/eval/test_eval"

    # do main for train eval & store it
    main(
        data_path=train_eval_data_path,
        save_report_path=f"{eval_data_path}/train_eval_report_01.json"
    )

    # do main for test eval & store it
    main(
        data_path=test_eval_data_path,
        save_report_path=f"{eval_data_path}/test_eval_report_01.json"
    )

"""
/mnt/Storage/ML/blender-5.1.0-linux-x64/blender --background --python src/model/evaluate.py
"""