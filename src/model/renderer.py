import json
from pathlib import Path
import bpy
import argparse
import gc
# from tqdm.auto import tqdm

import sys
import os
sys.path.insert(0, '/home/ML/TextureGeneration')

from src.data.txt_shader import TextShader

class Renderer:
    """
    Used for rendering shader text materials on a plane
    """
    def __init__(self) -> None:
        self.text_shader_converter = TextShader()
        self.current_mesh = None

    def purge_orphan_data(self) -> None:
        """
        Force unlinked data blocks out of memory to prevent RAM bloat in long loops.
        """
        for block in [bpy.data.meshes, bpy.data.materials, bpy.data.textures, bpy.data.cameras, bpy.data.lights]:
            for item in block:
                if item.users == 0:
                    block.remove(item)
        gc.collect()

    def scene_setup(self):
        """
        Sets camera, plane & lighting
        """
        # clears existing objects
        for obj in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        self.purge_orphan_data()

        # Adds camera
        cam_data = bpy.data.cameras.new('camera_data')
        cam = bpy.data.objects.new('camera', cam_data)
        bpy.context.scene.collection.objects.link(cam)
        bpy.context.scene.camera = cam
        cam.location = (0.0, 0.0, 2.75)
        cam.rotation_euler = (0, 0, 0)

        # Adds plane
        bpy.ops.mesh.primitive_plane_add()
        self.current_mesh = bpy.context.active_object
        self.current_mesh.name = "Render_Plane"

        # Adds Lighting
        if bpy.context.scene.world is not None:
            bpy.data.worlds.remove(bpy.context.scene.world, do_unlink=True)
    
        bpy.context.scene.world = bpy.data.worlds.new("World")
        bpy.context.scene.world.use_nodes = True
        bpy.context.scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.1, 0.1, 0.1, 1)

        bpy.ops.object.light_add(type='POINT', radius=0, align='WORLD', location=(0, 0, 2.78), scale=(1, 1, 1))
        bpy.data.objects['Point'].data.energy = 200

        

    def render_settings(
            self,
            engine : str = 'BLENDER_EEVEE',
            file_format : str = 'JPEG',
            resolution : int = 512
        ):
        """
        Sets the render settings config for rendering
        """
        scene = bpy.context.scene
        scene.render.engine = engine

        if engine == 'CYCLES':
            scene.cycles.samples = 128
            scene.cycles.use_denoising = False
            if bpy.context.preferences.addons['cycles'].preferences.get_devices():
                scene.cycles.device = 'GPU'
        
        scene.render.image_settings.file_format = file_format
        scene.render.image_settings.color_mode = 'RGB'
        scene.render.resolution_x = resolution
        scene.render.resolution_y = resolution

        scene.view_settings.view_transform = 'Standard'
        scene.view_settings.look = 'None'


    def add_material_to_mesh(
            self,
            material : bpy.types.Material = None
        ) -> None:
        """
        Adds material to specific mesh
        """
        if not self.current_mesh:
            raise ValueError("No mesh available to attach material to. Run scene_setup() first.")
        
        self.current_mesh.data.materials.clear()
        self.current_mesh.data.materials.append(material)
        

    def render(
            self,
            render_path : Path | str,
            text_shader : str = None,
            clean_material : bool = True
    ) -> tuple[bool, str | None]:
        """
        renders the scene
        Args:
            render_path : str - path to save rendered images
            text_shader : str - shader text for material

        Return:
            valid : bool - if the shader text is valid or not
            error : str - gives exact error if not valid else None
        """
        try:
            material = self.text_shader_converter.text_to_shader_graph(text_shader=text_shader)
        except Exception as e:
            return False, f"Shader Conversion Failed: {str(e)}"

        if not self.current_mesh or "Render_Plane" not in bpy.data.objects:
            self.scene_setup()
        
        self.add_material_to_mesh(material = material)

        save_path = Path(render_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        bpy.context.scene.render.filepath = str(render_path)
        bpy.ops.render.render(write_still=True)
        
        if clean_material:
            self.text_shader_converter.cleanup_material()
            self.current_mesh.data.materials.clear()
        self.purge_orphan_data()

        return True, None 

def main(
        result_json_path : str = None,
        save_json_path : str = None,
        render_path : str = None
) -> None:
    # scene setup
    renderer = Renderer()
    renderer.scene_setup()
    renderer.render_settings()

    # reading outputs from saved JSON & shader texts
    with open(result_json_path, "r") as f:
        results = json.load(f)

    result_info = {}
    count = 0
    shader_data_path = Path("ShaderDataset/val")
    for image, output in (results.items()):

        current_info = {}

        image = Path(image)
        img_parts = image.parts
        shader_text_path = shader_data_path.joinpath(*Path(image.with_suffix(".txt")).parts[3:])
        input_image_path = shader_data_path.joinpath(*img_parts[3:])

        current_info["shader_path"] = str(shader_text_path)
        current_info["output"] = output

        if not shader_text_path.exists():
            current_info["shader_text"] = None
            current_info["shader_error"] = f"File missing: {shader_text_path}"
            result_info[str(input_image_path)] = current_info
            continue

        with open(shader_text_path, "r") as f:
            text_shader = f.read()

        current_info["shader_text"] = text_shader

        # rendering both & saving in a rendered_images/ folder if valid 

        render_output_name = f"output_render_{image.stem}{image.suffix}"
        render_output_image = Path(render_path).joinpath(*img_parts[3:]).with_name(render_output_name)
        render_output_image.parent.mkdir(parents=True, exist_ok=True)

        render_shader_name = f"shader_render_{image.stem}{image.suffix}"
        render_shader_image = Path(render_path).joinpath(*img_parts[3:]).with_name(render_shader_name)
        render_shader_image.parent.mkdir(parents=True, exist_ok=True)

        output_valid, output_error = renderer.render(
            render_path = render_output_image,
            text_shader = output,
            clean_material=True
        )
        shader_valid, shader_error = renderer.render(
            render_path = render_shader_image,
            text_shader = text_shader,
            clean_material=True
        )

        # adding these paths to the results dict
        current_info["render_output"] = str(render_output_image) if output_valid else None
        current_info["render_shader"] = str(render_shader_image) if shader_valid else None

        # calculating LPIPS score if valid else log error & score MIN_VAL
        current_info["output_error"] = output_error
        current_info["shader_error"] = shader_error
        current_info["score"] = None 
        
        result_info[str(input_image_path)] = current_info
        
        count += 1
        if count == 100:
            break
    
    # save output_results dict into json
    with open(save_json_path, "w") as f:
        json.dump(result_info, f, indent=4)
    
    print(f"----- saved JSON at {save_json_path} -----")

if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_json_path",
        type=str,
        help="JSON file with all the outputs from inference"
    )
    parser.add_argument(
        "--save_json_path",
        type=str,
        help="path to save resulted output JSON"
    )
    parser.add_argument(
        "--render_path",
        type=str,
        help="path to save rendered samples"
    )
    args = parser.parse_args(argv)
    
    main(
        result_json_path=args.output_json_path,
        save_json_path=args.save_json_path,
        render_path=args.render_path
    )

""" 
/mnt/Storage/ML/blender-5.1.0-linux-x64/blender \
  --background \
  --python src/model/renderer.py \
  -- \
  --output_json_path results_new_tokens.json \
  --save_json_path results_new_tokens.json_RESULTS.json \
  --render_path RenderedOutputs/new_tokens
"""