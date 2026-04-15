import bpy

shader_nodes = {
    "OutputMaterial": 519300,
    "MixShader": 357961,
    "BsdfPrincipled": 554799,
    "ValToRGB": 578998,
    "TexNoise": 502939,
    "TexVoronoi": 265412,
    "Mapping": 446547,
    "TexCoord": 480333,
    "Bump": 477129,
    "BsdfGlossy": 80715,
    "MixRGB": 362983,
    "TexWave": 148634,
    "TexChecker": 23216,
    "Math": 123991,
    "Displacement": 50948,
    "HueSaturation": 13874,
    "Invert": 26111,
    "Emission": 20158,
    "TexMagic": 10075,
    "TexBrick": 155819,
    "TexMusgrave": 91043,
    "BrightContrast": 11789,
    "TexGradient": 16565,
    "BsdfGlass": 10474,
    "LayerWeight": 31613,
    "Fresnel": 26196,
    "SeparateXYZ": 4399,
    "VectorMath": 14701,
    "AmbientOcclusion": 9909,
    "NewGeometry": 20290,
    "RGBCurve": 17483,
    "Gamma": 5328,
    "LightPath": 11184,
    "BsdfTransparent": 4448,
    "CombineXYZ": 2450,
    "TexWhiteNoise": 553,
    "BsdfDiffuse": 13824,
    "BsdfTranslucent": 2493,
    "UVMap": 551,
    "NormalMap": 3874,
    "AddShader": 5369,
    "RGB": 11783,
    "MapRange": 4204,
    "Clamp": 514,
    "SubsurfaceScattering": 605,
    "Attribute": 365,
    "Normal": 622,
    "RGBToBW": 2262,
    "Wireframe": 1326,
    "Blackbody": 2344,
    "SeparateColor": 737,
    "VectorTransform": 329,
    "VectorRotate": 810,
    "ObjectInfo": 341,
    "CombineColor": 322,
    "BsdfRefraction": 220,
    "VolumeScatter": 314,
    "VolumeAbsorption": 278,
    "BsdfVelvet": 297,
    "VolumePrincipled": 60,
    "ShaderToRGB": 187,
    "CameraData": 77,
    "LightFalloff": 33,
    "VectorCurve": 55,
    "Tangent": 143,
    "BsdfAnisotropic": 231,
    "FloatCurve": 45,
    "Group": 2122,
    "VectorDisplacement": 18,
    "Bevel": 77,
    "Background": 16,
    "BsdfHair": 11,
    "Holdout": 11,
    "HairInfo": 11
}
AB_DICT = {
    "i" : "inputs",
    "o" : "outputs",
    "e" : "elements",
    "c" : "curves"
}

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
    

def build_shader_from_dsl(dsl_text, material_name="VLM_Generated_Material", ab_dict=AB_DICT):
    # 1. Setup Material
    mat = bpy.data.materials.new(name=material_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear() # Start with a clean slate

    # Split DSL into sections
    sections = dsl_text.split('\n')
    node_data = sections[0].replace('N|', '').split(';')
    prop_data = sections[1].replace('P|', '').split(';')
    link_data = sections[2].replace('L|', '').split(';')

    # Map to store variable names to actual Blender node objects
    node_map = {}

    # 2. Create Nodes (N|)
    for entry in node_data:
        var_name, node_type = entry.split(':')
        # Add the 'ShaderNode' prefix back during creation
        blender_type = f"ShaderNode{node_type}"
        node_map[var_name] = nodes.new(blender_type)
        node_map[var_name].name = var_name

    # 3. Set Properties (P|)
    for entry in prop_data:
        if not entry: continue

        #################################
        path, val = entry.split(':')
        
        path_parts = path.split('.')

        target_obj = node_map[path_parts[0]]
        current_val_attr = target_obj  

        for i, part in enumerate(path_parts):
            if i == 0 or i == (len(path_parts) - 1): continue
            if part[1:].isdigit():
                attr_name = ab_dict[part[0]]
                attr_list = getattr(current_val_attr, attr_name)
                current_val_attr = attr_list[int(part[1:])]
            else:
                current_val_attr = getattr(current_val_attr, part)
              
        final_part = path_parts[-1]

        if final_part[1:].isdigit():
            val_attr = getattr(current_val_attr, ab_dict[final_part[0]])[int(final_part[1:])]
            setattr(val_attr, "default_value", eval(val))
#            val_attr = eval(val)
        else:
            setattr(current_val_attr, final_part, eval(val))

        #################################
        # path, val = entry.split(':')
        # node_var = path.split('.')[0]
        # attr_chain = path.split('.')[1:]
        
        # target_obj = node_map[node_var]
        
        # # Resolve the minified path back to Blender objects
        # # e.g., i4 -> inputs[4], e0 -> elements[0]
        # for part in attr_chain:
        #     if part.startswith('i'):
        #         idx = int(part[1:])
        #         target_obj = target_obj.inputs[idx].default_value
        #     elif part.startswith('e'):
        #         idx = int(part[1:])
        #         target_obj = target_obj.color_ramp.elements[idx]
        #     else:
        #         # Handle direct attributes like .operation
        #         # Use exec/setattr carefully or a simple mapping
        #         setattr(target_obj, part, eval(val))

    # 4. Create Links (L|)
    for entry in link_data:
        if not entry: continue
        out_part, in_part = entry.split('>')
        out_node, out_idx = out_part.split('.')
        in_node, in_idx = in_part.split('.')
        
        links.new(node_map[out_node].outputs[int(out_idx)], 
                  node_map[in_node].inputs[int(in_idx)])

    return mat


text = "N|material_output:OutputMaterial;principled_bsdf:BsdfPrincipled;colorramp:ValToRGB;voronoi_texture:TexVoronoi;mapping:Mapping;texture_coordinate:TexCoord;displacement:Displacement\nP|principled_bsdf.i3:[0.54, 0.55, 0.43, 0.58];principled_bsdf.i4:1.49;principled_bsdf.i7:1.23;principled_bsdf.i8:0.603;principled_bsdf.i9:0.548;principled_bsdf.i10:0.536;principled_bsdf.i11:0.292;principled_bsdf.i14:0.77;principled_bsdf.i19:[0.01, 0.0, 0.0, 0.69];principled_bsdf.i20:1.07;colorramp.color_ramp.interpolation:'CONSTANT';colorramp.color_ramp.e0.position:0.026;colorramp.color_ramp.e0.color:[0.3, 0.29, 0.46, 0.53];colorramp.color_ramp.e1.position:0.461;colorramp.color_ramp.e1.color:[0.61, 0.56, 0.27, 0.91];voronoi_texture.i1:0.015;voronoi_texture.i2:15.7;voronoi_texture.i3:0.016;voronoi_texture.i5:0.006;voronoi_texture.feature:'SMOOTH_F1';displacement.i1:0.006;displacement.i2:0.167;displacement.space:'WORLD'\nL|principled_bsdf.0>material_output.0;displacement.0>material_output.2;colorramp.0>principled_bsdf.0;voronoi_texture.1>colorramp.0;mapping.0>voronoi_texture.0;texture_coordinate.2>mapping.0;voronoi_texture.1>displacement.0"

plane = setup_scene()
setup_generalized_lighting()

mat = build_shader_from_dsl(text)

plane.data.materials.append(mat)