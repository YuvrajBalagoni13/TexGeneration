import bpy
import ast 
import json
from typing import Tuple, Optional

class DSLShaders:
    def __init__(
            self, 
            nodes:list[str] = None, 
            properties: list[str] = None, 
            links:list[str] = None,
            dsl_text:str = None
            ) -> None:
        """
        Base Shader class with basic functionalities
        """

        self.nodes_info = nodes or []
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
        self.available_node_types = [name[10:] for name in dir(bpy.types) if name.startswith("ShaderNode")]
        self.temp_mat = None
        self._node_type_cache = {}  
        self._last_validation_result = None

        try:
            with open("JSON_files/nodes_data_51.json", "r") as f:
                self.check_valid_inputs = json.load(f)
        except:
            raise KeyError(f"JSON file not found.")

    def reset(self) -> None:
        """
        resets all the infos
        """
        self.nodes_info = []
        self.properties_info = []
        self.links_info = []
        self.dsl_text = ""
        self.current_node_dict = {}
        self.current_node_mapping = {}
        self._last_validation_result = None

    def get_txt(self) -> str:
        """
        joins node, properties, links info to get dsl_text
        """
        if not self.dsl_text:
            self.dsl_text = f"N|{';'.join(self.nodes_info)}\nP|{';'.join(self.properties_info)}\nL|{';'.join(self.links_info)}"
        return self.dsl_text

    def setup_material(self):
        """
        Creates a single reusable material for validation
        """
        if self.temp_mat is None:
            self.temp_mat = bpy.data.materials.new(name="VALIDATION_TEMP")
            self.temp_mat.use_nodes = True
            self.temp_mat.node_tree.nodes.clear()
        
    def cleanup_material(self):
        """
        Clears nodes but keeps material for reuse
        """
        if self.temp_mat and self.temp_mat.node_tree:
            self.temp_mat.node_tree.nodes.clear()
        self.current_node_dict.clear()
        
    def destroy_material(self):
        """
        Completely remove material when done with all conversions
        """
        if self.temp_mat:
            bpy.data.materials.remove(self.temp_mat)
            self.temp_mat = None

    def save_txt(self, text_file_path: str) -> None:
        """
        Saves the DSL to a text file.
        """
        _ = self.get_txt()
        
        # Save the file
        with open(text_file_path, "w") as f:
            f.write(self.dsl_text)


class ConvertCodeToDSL:
    """
    Main class for converting python code into DSL
    """
    def __init__(self) -> None:
        with open("JSON_files/nodes_data_36.json", "r") as f:
            self.nodes_data_v36 = json.load(f)

        with open("JSON_files/nodes_data_51.json", "r") as f:
            self.nodes_data_v51 = json.load(f)

        self.dsl_shader = DSLShaders()
        self._full_path_cache = {}  

    def get_full_path(self, node, use_cache: bool = True) -> str:
        """
        Cached version of path extraction
        """
        if use_cache:
            cache_key = str(id(node))
            if cache_key in self._full_path_cache:
                return self._full_path_cache[cache_key]
        
        result = self._get_full_path_impl(node)
        if use_cache:
            self._full_path_cache[cache_key] = result
        return result
    
    def _get_full_path_impl(self, node) -> str:
        """
        Actual implementation without caching
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            base = self.get_full_path(node.value)
            return f"{base}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            base = self.get_full_path(node.value)
            base_name_list = base.split(".")
            if base_name_list:
                base_name_list[-1] = base_name_list[-1][0] if base_name_list[-1] else ""
                base = ".".join(base_name_list)
            
            if isinstance(node.slice, ast.Constant):
                index = node.slice.value
            elif isinstance(node.slice, ast.Index):
                index = node.slice.value.value if hasattr(node.slice.value, 'value') else node.slice.value
            else:
                index = "x"

            if base_name_list and base_name_list[-1] and base_name_list[-1][0] == "i":
                if base_name_list[0] in self.current_node_vartype_mapping:
                    node_type = self.current_node_vartype_mapping[base_name_list[0]]
                    if node_type in self.nodes_data_v36:
                        attr_name = self.nodes_data_v36[node_type]["inputs"][index]
                        if attr_name not in self.nodes_data_v51[node_type]["inputs"]:
                            raise ValueError(f"{attr_name} not as input to {node_type}.")
                        return f"{base}-{attr_name}"
            
            return f"{base}{index}"
        return ""

    def convert(self, python_path:str, text_path:str = None) -> str:
        """
        file conversion
        """
        # Reuse material across conversions
        self.dsl_shader.reset()
        self.dsl_shader.cleanup_material()

        self.dsl_shader.setup_material()
        nodes = self.dsl_shader.temp_mat.node_tree.nodes
        self.current_node_vartype_mapping = {}
        self._full_path_cache.clear()  

        with open(python_path, "r") as f:
            tree = ast.parse(f.read())

        skip_first_mat = True
        
        for node in ast.walk(tree):
            # Nodes
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                if getattr(node.value.func, 'attr', None) == 'new':
                    if isinstance(node.targets[0], ast.Name):
                        var_name = node.targets[0].id
                        node_type = node.value.args[0].value[10:]
                        
                        if node_type not in self.dsl_shader.available_node_types:
                            raise ValueError(f"{var_name} : {node_type} not available")
                        
                        if node_type == "Group":
                            raise ValueError(f"found ShaderNodeGroup so skipping it ........")
                        
                        self.dsl_shader.nodes_info.append(f"{var_name}:{node_type}")
                        self.dsl_shader.current_node_dict[var_name] = nodes.new(f"ShaderNode{node_type}")
                        self.current_node_vartype_mapping[var_name] = f"ShaderNode{node_type}"

            # Properties
            elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Attribute):
                if skip_first_mat:
                    skip_first_mat = False
                    continue

                target = node.targets[0]
                try:     
                    path = self.get_full_path(target)
                    path_list = path.split(".")
                    
                    if path:  
                        val = ast.literal_eval(node.value)

                    if path_list[-1] == "default_value":
                        path_list[-1] = "dv"
                        path = ".".join(path_list)
                        if path_list[-2].split("-")[-1] in ["Specular Tint", "Sheen Tint"]:
                            val = [val, val, val, 1]
                        
                    # Format value
                    if isinstance(val, str):
                        val = f"'{val}'"
                    elif isinstance(val, float):
                        val = round(val, 3)
                    elif isinstance(val, list):
                        val = [round(v, 3) if isinstance(v, float) else v for v in val]
                        
                    self.dsl_shader.properties_info.append(f"{path}:{val}")
                except Exception as e:
                    pass
                
            # Links
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                if getattr(node.value.func, 'attr', None) == 'new':
                    if len(node.value.args) == 2:
                        out_part = node.value.args[0]
                        in_part = node.value.args[1]
                        
                        if isinstance(out_part, ast.Subscript) and isinstance(in_part, ast.Subscript):
                            try:
                                out_var = out_part.value.value.id
                                out_idx = out_part.slice.value
                                in_var = in_part.value.value.id
                                in_idx = in_part.slice.value
                                
                                if out_var in self.current_node_vartype_mapping:
                                    out_node_type, in_node_type = self.current_node_vartype_mapping[out_var], self.current_node_vartype_mapping[in_var]
                                    out_socket_name = self.nodes_data_v36[out_node_type]["outputs"][out_idx]
                                    in_socket_name = self.nodes_data_v36[in_node_type]["inputs"][in_idx]
                                    if out_socket_name not in self.nodes_data_v51[out_node_type]["outputs"]:
                                        raise ValueError(f"{out_socket_name} does not exist as output to {out_node_type}")
                                    if in_socket_name not in self.nodes_data_v51[in_node_type]["inputs"]:
                                        raise ValueError(f"{in_socket_name} does not exist as input to {in_node_type}")
                                    self.dsl_shader.links_info.append(f"{out_var}.{out_socket_name}>{in_var}.{in_socket_name}")
                            except Exception as e:
                                print(f"Link error: {e}")
                                continue
                    else:
                        try:
                            target = node.value.func
                            path = self.get_full_path(target)

                            if len(node.value.args) == 1:
                                val = node.value.args[0].value
                            elif len(node.value.args) > 2:
                                val = []
                                for i in len(node.value.args):
                                    val.append(node.value.args[i].value)

                            # Format value
                            if isinstance(val, str):
                                val = f"'{val}'"
                            elif isinstance(val, float):
                                val = round(val, 3)
                            elif isinstance(val, list):
                                val = [round(v, 3) if isinstance(v, float) else v for v in val]
                                
                            self.dsl_shader.properties_info.append(f"{path}:{val}")
                        except:
                            raise ValueError(f"Not able to create new for part of node.")


        text = self.dsl_shader.get_txt()

        if text_path:
            self.dsl_shader.save_txt(text_path)

        
        self.dsl_shader.reset()
        self.dsl_shader.cleanup_material()  
        return text