# src/model/blender_render.py
import bpy
import json
import os
import sys
from pathlib import Path

# add only your local src to path (no torch/numpy needed here)
sys.path.insert(0, os.getcwd())
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

def main(inference_results_path, render_output_path, report_path):
    with open(inference_results_path, "r") as f:
        results = json.load(f)

    txt_shader = TextShader()
    plane = setup_scene()
    setup_lighting()
    set_render_settings()

    render_results = []

    for i, sample in enumerate(results):
        log_error = None
        render_image_path = None

        try:
            material = txt_shader.text_to_shader_graph(
                text_shader=sample["output_shader"],
                material_name=Path(sample["image_path"]).stem
            )
            plane.data.materials.clear()
            plane.data.materials.append(material)

            render_image_path = os.path.join(
                render_output_path,
                f"render_{Path(sample['image_path']).stem}.png"
            )
            bpy.context.scene.render.filepath = render_image_path
            bpy.ops.render.render(write_still=True)

        except Exception as e:
            log_error = f"Error: {e}"

        render_results.append({
            **sample,
            "render_path": str(render_image_path),
            "error": log_error
        })

    with open(report_path, "w") as f:
        json.dump(render_results, f, indent=4)

    print(f"Rendering done. Report saved to {report_path}")

if __name__ == "__main__":
    main(
        inference_results_path="Dataset/eval/inference_results.json",
        render_output_path="Dataset/eval/renders",
        report_path="Dataset/eval/render_report.json"
    )