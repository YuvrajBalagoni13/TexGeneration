import bpy
import json 
import ast
from collections import defaultdict
from tqdm.auto import tqdm
from pathlib import Path

class TextShader:
    def __init__(self, nodes:list[str] = None, properties: list[str] = None, links:list[str] = None,dsl_text:str = None) -> None:
        self.node_info = nodes or []
        self.properties_info = properties or []
        self.links_info = links or []
        self.dsl_text = dsl_text or ""
        self.current_node_dict = {}
        self.current_node_mapping = {}
        self.subscript_dict = {
            "i" : "inputs",
            "o" : "outputs",
            "e" : "elements",
            "c" : "curves",
            "p" : "points"
        }
        self.temp_mat = None
        try:
            with open("JSON_files/nodes_data_51.json", "r") as f:
                self.check_valid_inputs = json.load(f)
        except:
            raise KeyError(f"JSON file not found.")

    def _reset_info(self) -> None:
        self.node_info = []
        self.properties_info = []
        self.links_info = []
        self.current_node_dict = {}
        self.current_node_mapping = {}

    def _reset_shader(self) -> None:
        self._reset_info()
        self.dsl_text = ""

    def setup_material(self, material_name : str = "VALIDATION_TEMP") -> None:
        self.temp_mat = None
        self.temp_mat = bpy.data.materials.new(name=material_name)
        self.temp_mat.use_nodes = True
        self.temp_mat.node_tree.nodes.clear()

    def _parse_txt_shader(self) -> None:
        self._reset_info()
        lines = self.dsl_text.strip().split("\n")
        if len(lines) != 3: raise ValueError(f"Invalid Format: {len(lines)} total lines != 3 lines for nodes, properties & links.")

        for line in lines:
            if line.startswith("N|"):
                self.node_info = [n for n in line[2:].split(";") if n]
            elif line.startswith("P|"):
                self.properties_info = [p for p in line[2:].split(";") if p]
            elif line.startswith("L|"):
                self.links_info = [l for l in line[2:].split(";") if l]

    def get_node_properties(self, node) -> str:
        """
        gives all the none input properties.
        """
        properties = []
        
        i = 0
        for prop in node.bl_rna.properties:
            if not prop.is_readonly and i > 33:
                attr_name = prop.identifier
                value = getattr(node, attr_name)
                
                if hasattr(value, "to_list"):
                    value = value.to_list()
                elif hasattr(value, "hsv"):
                    value = [value[i] for i in range(len(value))]
                    
                properties.append(f"{attr_name}:{value}")
            i += 1
        return ",".join(properties)

    def text_to_shader_graph(self, text_shader_path : str = None, text_shader : str = None, material_name: str = "TemporaryMaterial") -> bpy.types.Material:
        if text_shader_path:
            with open(text_shader_path, "r") as f:
                self.dsl_text = f.read()

        if self.dsl_text is None and text_shader is None:
            raise ValueError(f"No text shader info given.")
        
        self.dsl_text = self.dsl_text or text_shader
        self._parse_txt_shader()

        self.setup_material(material_name = material_name)

        for node_info in self.node_info:
            var, node_type = node_info.split(":")
            node_type = f"ShaderNode{node_type}"

            if node_type not in self.check_valid_inputs:
                raise ValueError(f"{node_type} does not exist in latest Blender version 5.1")

            self.current_node_dict[var] = self.temp_mat.node_tree.nodes.new(node_type)
            self.current_node_mapping[var] = node_type

        for key, val in self.current_node_mapping.items():
            if val == "ShaderNodeTexVoronoi":
                self.current_node_dict[key].voronoi_dimensions = '4D'
                self.current_node_dict[key].distance = 'MINKOWSKI'
            elif val == "ShaderNodeTexNoise":
                self.current_node_dict[key].noise_dimensions = '4D'

        for properties in self.properties_info:
            property_path, val = properties.split(":")

            path_list = property_path.split(".")

            if path_list[0] not in self.current_node_mapping:
                raise ValueError(f"Incorrect node name : {path_list[0]} not in current nodes added via text.")
            
            current_attr = self.current_node_dict[path_list[0]]
            current_node_type = self.current_node_mapping[path_list[0]]

            evaluated_val = ast.literal_eval(val)

            if type(evaluated_val) == str and evaluated_val == 'RANDOM_WALK_FIXED_RADIUS':
                evaluated_val = 'RANDOM_WALK_SKIN'

            if type(evaluated_val) == str and evaluated_val == 'SHARP':
                evaluated_val = 'GGX'


            for path in path_list[1:-1]:
                if path.startswith("i-"):
                    socket_name = path[2:]
                    if socket_name not in self.check_valid_inputs[current_node_type]["inputs"]:
                        raise ValueError(f"{socket_name} not present as inputs in {current_node_type}")
                    try:
                        if socket_name in current_attr.inputs:
                            current_attr = current_attr.inputs[socket_name]
                        else:
                            target_socket = next((s for s in current_attr.inputs if s.identifier == socket_name), None)
                            
                            if target_socket:
                                current_attr = target_socket
                            else:
                                raise ValueError(f"'{socket_name}' not found as Name or Identifier in {current_node_type}")
                    except Exception as e:
                       raise ValueError(f"Error accessing socket '{socket_name}' on {path_list[0]}: {e}")
                
                elif path[0] in self.subscript_dict and path[1:].isdigit():
                    attr_name = self.subscript_dict[path[0]]
                    idx = int(path[1:])
                    collection = getattr(current_attr, attr_name)
                    if idx > len(collection):

                        raise ValueError(f"{idx} is out of range for {attr_name} in {current_node_type}")
                    current_attr = collection[idx]

                else:
                    if not hasattr(current_attr, path):
                        raise ValueError(f"{path} is not an attribute to {current_attr}")
                    if path == "dv":
                        continue
                    current_attr = getattr(current_attr, path)
            try:
                if path_list[-1] == "dv":
                    current_attr.default_value = evaluated_val
                elif path_list[-1] == "new":
                    current_attr.new(evaluated_val)
                else:
                    current_attr = setattr(current_attr, path_list[-1], evaluated_val)
            except Exception as e:
                if path_list[-1] == "dv":
                    socket_name = path_list[-2].split("-")[-1]
                else:
                    socket_name = path_list[-1]
                raise ValueError(f"{current_node_type}'s input socket {socket_name} does not take value {evaluated_val}")
        
        for links in self.links_info:
            out_part, in_part = links.split(">")
            out_node, out_socket = out_part.split(".")
            in_node, in_socket = in_part.split(".")

            if out_node not in self.current_node_mapping:
                raise ValueError(f"Incorrect node name : {out_node} not in current nodes added via text based shader.")
            if in_node not in self.current_node_mapping:
                raise ValueError(f"Incorrect node name : {in_node} not in current nodes added via text based shader.")
            
            out_shader_node = self.current_node_dict[out_node]
            in_shader_node = self.current_node_dict[in_node]

            if out_socket not in out_shader_node.outputs:
                raise ValueError(f"{out_socket} not available as output in {self.current_node_mapping[out_node]} when {self.get_node_properties(out_shader_node)}")
            if in_socket not in in_shader_node.inputs:
                raise ValueError(f"{in_socket} not available as input in {self.current_node_mapping[in_node]} when {self.get_node_properties(in_shader_node)}")
        
            self.temp_mat.node_tree.links.new(out_shader_node.outputs[out_socket], in_shader_node.inputs[in_socket])

        return self.temp_mat

    def shader_graph_to_text(self) -> None:
        pass


if __name__ == "__main__":
    unique_error_logs = defaultdict(list)
    dataset_path = Path("Dataset/infinigen")
    txt_files = list(dataset_path.rglob("*.txt"))
    for files in tqdm(txt_files):
        try:
            txt_shader = TextShader()

            material = txt_shader.text_to_shader_graph(text_shader_path=files)
            material.node_tree.nodes.clear()
        except Exception as e:
            print(e, files)
            unique_error_logs[f"{e}"].append(str(files))

    with open("JSON_files/error_logs.json", "w") as f:
        json.dump(unique_error_logs, f, indent=4)

    print("logged errors.")

"""
/mnt/Storage/ML/blender-5.1.0-linux-x64/blender --background --python src/data/txt_shader.py
"""