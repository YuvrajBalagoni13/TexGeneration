import json
from pathlib import Path
import bpy
import argparse

import sys
import os
sys.path.insert(0, '/home/ML/TextureGeneration')

from src.data.txt_shader import TextShader

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

    bpy.context.scene.render.filepath = str(render_image_path)
    bpy.ops.render.render(write_still=True)
    return render_image_path

def main(eval_json_path: str = None) -> None:
    with open(eval_json_path, "r") as f:
        responses = json.load(f)

    plane = setup_scene()
    setup_generalized_lighting()
    set_render_settings()

    txt_shader = TextShader()
    for k, v in responses.items():
        responses[k]["error"] = None
        try:
            material = txt_shader.text_to_shader_graph(text_shader=v["output_shader"])
            print("----- material good -----")

            render_image_path = add_material_to_plane(
                plane = plane, 
                material = material, 
                image_path = v["image_path"]
                )
            responses[k]["render_path"] = str(render_image_path)
            print("----- rendered -----")
            
            material.node_tree.nodes.clear()
        except Exception as e:
            responses[k]["error"] = f"Error: {e}"
    print("----- done -----")
    
    with open(eval_json_path, "w") as f:
        json.dump(responses, f, indent=4)
    print("----- saved -----")

if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save_json_path",
        required=True
    )

    args = parser.parse_args(argv)

    main(args.save_json_path)

""" 
/mnt/Storage/ML/blender-5.1.0-linux-x64/blender --background --python src/model/blender_render.py
"""