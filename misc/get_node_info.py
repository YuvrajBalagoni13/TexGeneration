# shader_nodes = [name[10:] for name in dir(bpy.types) if name.startswith("ShaderNode")]

import bpy
import json

def get_all_node_sockets():
    # 1. Create a temporary material and node tree
    temp_mat = bpy.data.materials.new(name="Socket_Scraper")
    temp_mat.use_nodes = True
    nodes = temp_mat.node_tree.nodes
    
    # 2. Get all ShaderNode types from bpy.types
    node_types = [n for n in dir(bpy.types) if n.startswith("ShaderNode")]
    
    node_data = {}

    for nt in node_types:
        try:
            # Create a temporary node of this type
            node = nodes.new(nt)
            node_data[nt] = {
                "inputs" : [],
                "outputs" : []
            }
            
            # Scrape Inputs and Outputs
            for i, input_name in enumerate(node.inputs):
                node_data[nt]["inputs"].append(input_name.identifier)
            for i, output_name in enumerate(node.outputs):
                node_data[nt]["outputs"].append(output_name.identifier)
            
            # Remove the node to keep memory clean
            nodes.remove(node)
            
        except Exception as e:
            print(f"Error : {e}")
            # Some classes in bpy.types are base classes and cannot be created
            continue

    # 3. Cleanup: Remove the temporary material
    bpy.data.materials.remove(temp_mat)
    return node_data

# Run and Print
data = get_all_node_sockets()
with open("nodes_data_36.json", "w") as f:
    json.dump(data, f, indent=4)

"""
/mnt/Storage/ML/blender-3.6.23-linux-x64/blender --background --python get_node_info.py
"""