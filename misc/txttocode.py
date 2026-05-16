import bpy

AB_DICT = {
    "i" : "inputs",
    "e" : "elements",
    "c" : "curves"
}

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