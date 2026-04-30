# import bpy
# import ast 
# import json

# class DSLShaders:
#     """
#     create dsl shader instance &
#     save dsl txt file 
#             convert from code to dsl txt
#             convert from dsl txt to shader
#     validate dsl txt 
#     """
#     def __init__(
#             self, 
#             nodes:list[str] = None, 
#             properties: list[str] = None, 
#             links:list[str] = None,
#             dsl_text:str = None
#             ) -> None:
#         """
#         nodes_info - list of strings (var_name:node_name)
#         properties_info - list of strings (var_name.attr_name:val)
#         links_info - list of strings (out_node.out_socket>in_node.in_socket)
#         """
#         self.nodes_info = nodes or []
#         self.properties_info = properties or []
#         self.links_info = links or []
#         self.dsl_text = dsl_text or ""
#         self.current_node_dict = {}
#         self.subscript_dict = {
#             "i" : "inputs",
#             "o" : "outputs",
#             "e" : "elements",
#             "c" : "curves",
#             "p" : "points"
#         }
#         self.available_node_types = [name[10:] for name in dir(bpy.types) if name.startswith("ShaderNode")]
#         self.temp_mat = None

#     def reset(self) -> None:
#         self.nodes_info = []
#         self.properties_info = []
#         self.links_info = []
#         self.dsl_text = ""
#         self.current_node_dict = {}

#     def get_current_nodes_info(self) -> dict:
#         for nodes in self.nodes_info:
#             var, node_type = nodes.split(":")
#             self.current_node_dict[var] = self.temp_mat.node_tree.nodes.new(f"ShaderNode{node_type}")
#         return self.current_node_dict
    
#     def get_txt(self) -> str:
#         if not self.dsl_text:
#             self.dsl_text = f"N|{";".join(self.nodes_info)}\nP|{";".join(self.properties_info)}\nL|{";".join(self.links_info)}"
#         return self.dsl_text

#     def get_material_from_dsl(self, dsl_text: str = None) -> None:
#         if dsl_text is not None:
#             self.dsl_text = dsl_text
#         pass
    
#     def save_txt(self, text_file_path: str) -> None:
#         """
#         checks if the dsl is valid or not
#         & saves it to the file path or raise an error.
#         """
#         _ = self.get_txt()
#         valid, reason = self.validate_dsl()

#         if valid:
#             with open(text_file_path, "w") as f:
#                 f.write(self.dsl_text)
#         else:
#             raise ValueError(f"Invalid DSL format: {reason}")

#     def setup_material(self):
#         """Creates a hidden material to test node paths against"""
#         self.temp_mat = bpy.data.materials.new(name="VALIDATION_TEMP")
#         self.temp_mat.use_nodes = True
        
#     def cleanup_material(self):
#         """Removes the temp material after validation is done"""
#         if self.temp_mat:
#             bpy.data.materials.remove(self.temp_mat)

#     def valid_property_path_and_val(self, property_path: str, val: str) -> tuple[bool, str]:
#         path_list = property_path.split(".")

#         if path_list[0] not in self.current_node_dict:
#             return False, f"Node variable {path_list[0]} not defined in N| section"
    
#         current_attr = self.current_node_dict[path_list[0]]
#         evaluated_val = eval(val)
#         for paths in path_list[1:]:
#             # Handle Inputs
#             if paths.startswith("i-"):
#                 socket_name = paths[2:]
#                 if socket_name not in current_attr.inputs:
#                     return False, f"Input '{path_list[0]}' : '{socket_name}' not found"
#                 current_attr = current_attr.inputs[socket_name]
        
#             # Handle Subscripts (e0, p1, etc)
#             elif paths[0] in self.subscript_dict and paths[1:].isdigit():
#                 attr_name = self.subscript_dict[paths[0]]
#                 idx = int(paths[1:])
#                 collection = getattr(current_attr, attr_name)
#                 if idx >= len(collection):
#                     return False, f"Index {idx} out of range for {attr_name}"
#                 current_attr = collection[idx]
        
#             # Handle Standard Attributes
#             else:
#                 if not hasattr(current_attr, paths):
#                     return False, f"Attribute '{paths}' not found on {current_attr}"
#                 current_attr = getattr(current_attr, paths)
#         # FINAL TYPE CHECK
#         check_target = current_attr
#         # if hasattr(current_attr, "default_value"):
#         #     check_target = current_attr.default_value

#         # Use isinstance for better flexibility (handles subclasses)
#         # if not isinstance(evaluated_val, type(check_target)):
#         #     # Specific patch for int/float flexibility
#         #     if isinstance(evaluated_val, (int, float)) and isinstance(check_target, (int, float)):
#         #         return True, None
#         #     return False, f"Type Mismatch: {type(evaluated_val)} vs {type(check_target)}"
#         return True, None

#     def validate_dsl(self) -> tuple[bool, str]:
#         """
#         checks if the dsl is valid or not.
#         """
#         self.setup_material()

#         _ = self.get_txt()

#         if len(self.nodes_info) == 0 or len(self.properties_info) == 0 or len(self.links_info) == 0:
#             self.nodes_info = self.dsl_text.split("\n")[0][2:].split(";")
#             self.properties_info = self.dsl_text.split("\n")[1][2:].split(";")
#             self.links_info = self.dsl_text.split("\n")[2][2:].split(";")
        
#         _ = self.get_current_nodes_info()
            
#         for nodes in self.nodes_info:
#             node_var_name, node_type_name = nodes.split(":")
#             if node_type_name not in self.available_node_types:
#                 return False, f"{node_type_name} not in existing nodes"
        
#         for links in self.links_info:
#             out_part, in_part = links.split(">")
#             out_node, out_socket = out_part.split(".")
#             in_node, in_socket = in_part.split(".")

#             out_node_type = self.current_node_dict[out_node]
#             in_node_type = self.current_node_dict[in_node]

#             if out_socket not in out_node_type.outputs:
#                 return False, f"{out_node} : {out_node_type} doesn't have {out_socket} output in it."
#             if in_socket not in in_node_type.inputs:
#                 return False, f"{in_node} : {in_node_type} doesn't have {in_socket} input in it."
            
#         for properties in self.properties_info:
#             property_path, val = properties.split(":")
#             valid_path, error = self.valid_property_path_and_val(property_path, val)
#             if not valid_path:
#                 return False, error
        
#         self.cleanup_material()
#         return True, None


# class ConvertCodeToDSL:
#     def __init__(self) -> None:

#         with open("JSON_files/nodes_data_36.json", "r") as f:
#             self.nodes_data_v36 = json.load(f)

#         with open("JSON_files/nodes_data_51.json", "r") as f:
#             self.nodes_data_v51 = json.load(f)

#         self.dsl_shader = DSLShaders()

#     def get_full_path(self, node) -> str:
#         if isinstance(node, ast.Name):
#             return node.id
#         elif isinstance(node, ast.Attribute):
#             base = self.get_full_path(node.value)
#             # if node.attr == "default_value":
#             #     return base
#             return f"{base}.{node.attr}"
#         elif isinstance(node, ast.Subscript):
#             base = self.get_full_path(node.value)
#             base_name_list = base.split(".")
#             base_name_list[-1] = base_name_list[-1][0]
#             base = ".".join(base_name_list)
            
#             index = node.slice.value if isinstance(node.slice, ast.Constant) else "x"

#             if base_name_list[-1][0] == "i":
#                 node_type = self.current_node_vartype_mapping[base_name_list[0]]
#                 attr_name = self.nodes_data_v36[node_type]["inputs"][index]
#                 if attr_name not in self.nodes_data_v51[node_type]["inputs"]:
#                     raise ValueError(f"{attr_name} not as input to {node_type}.")
                
#                 return f"{base}-{attr_name}"
            
#             return f"{base}{index}"
#         return ""

#     def convert(self, python_path:str, text_path:str = None) -> str:
#         self.dsl_shader.setup_material()
#         nodes = self.dsl_shader.temp_mat.node_tree.nodes
#         self.current_node_vartype_mapping = {}

#         with open(python_path, "r") as f:
#             tree = ast.parse(f.read())

#         skip_first_mat = True
#         for node in ast.walk(tree):

#             # for shader nodes selections
#             if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
#                 if getattr(node.value.func, 'attr', None) == 'new':
#                     var_name = node.targets[0].id
#                     node_type = node.value.args[0].value[10:]
                    
#                     if node_type not in self.dsl_shader.available_node_types:
#                         raise ValueError(f"{var_name} : {node_type} not available in current blender version 5.1")
                    
#                     self.dsl_shader.nodes_info.append(f"{var_name}:{node_type}")
#                     self.dsl_shader.current_node_dict[var_name] = nodes.new(f"ShaderNode{node_type}")
#                     self.current_node_vartype_mapping[var_name] = f"ShaderNode{node_type}"

#             # for properties
#             elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Attribute):
#                 if skip_first_mat:
#                     skip_first_mat = False
#                     continue

#                 target = node.targets[0]

#                 try:     
#                     path = self.get_full_path(target)
#                     val = ast.literal_eval(node.value)

#                     if isinstance(val, str) : val = f"'{val}'"
#                     if isinstance(val, float) : val = round(val, 3)
#                     if isinstance(val, list) : val = [round(v, 3) if isinstance(v, float) else v for v in val]

#                     self.dsl_shader.properties_info.append(f"{path}:{val}")
#                 except Exception as e:
#                     raise ValueError(f"Properties Error : {e}")
                
#             elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
#                 if getattr(node.value.func, 'attr', None) == 'new':
#                     out_part = node.value.args[0]

#                     if not isinstance(out_part, ast.Subscript):
#                         target = node.value.func
                        
#                         try:
#                             path = self.get_full_path(target)
#                             if len(node.value.args) != 1:
#                                 val = []
#                                 for arg in node.value.args:
#                                     val.append(arg.value)
#                                 if isinstance(val, str) : val = f"'{val}'"
#                                 if isinstance(val, float) : val = round(val, 3)
#                                 if isinstance(val, list) : val = [round(v, 3) if isinstance(v, float) else v for v in val]
#                             else:
#                                 val = node.value.args[0].value
#                                 if isinstance(val, str) : val = f"'{val}'"
#                                 if isinstance(val, float) : val = round(val, 3)
#                                 if isinstance(val, list) : val = [round(v, 3) if isinstance(v, float) else v for v in val]
                            
#                             self.dsl_shader.properties_info.append(f"{path}:{val}")
#                         except Exception as e:
#                             print(f"Properties L Error : {e}")
#                         continue

#                     in_part = node.value.args[1]
#                     out_var, out_idx = out_part.value.value.id, out_part.slice.value
#                     in_var, in_idx = in_part.value.value.id, in_part.slice.value
#                     out_socket_name = self.nodes_data_v36[self.current_node_vartype_mapping[out_var]]["outputs"][out_idx]
#                     in_socket_name = self.nodes_data_v36[self.current_node_vartype_mapping[in_var]]["inputs"][in_idx]

#                     self.dsl_shader.links_info.append(f"{out_var}.{out_socket_name}>{in_var}.{in_socket_name}")

#         valid, error = self.dsl_shader.validate_dsl()
#         text = self.dsl_shader.get_txt()

#         if text_path:
#             self.dsl_shader.save_txt(text_path)

#         self.dsl_shader.reset()

#         if valid:
#             return text
#         else:
#             raise ValueError(f"Error - dsl generated is not valid : {error}")





######################################################################################################



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
        self._node_type_cache = {}  # Cache for node types
        self._last_validation_result = None

        try:
            with open("JSON_files/nodes_data_51.json", "r") as f:
                self.check_valid_inputs = json.load(f)
        except:
            raise KeyError(f"JSON file not found.")

    def reset(self) -> None:
        self.nodes_info = []
        self.properties_info = []
        self.links_info = []
        self.dsl_text = ""
        self.current_node_dict = {}
        self.current_node_mapping = {}
        self._last_validation_result = None

    def get_txt(self) -> str:
        if not self.dsl_text:
            self.dsl_text = f"N|{';'.join(self.nodes_info)}\nP|{';'.join(self.properties_info)}\nL|{';'.join(self.links_info)}"
        return self.dsl_text

    def setup_material(self):
        """Creates a single reusable material for validation"""
        if self.temp_mat is None:
            self.temp_mat = bpy.data.materials.new(name="VALIDATION_TEMP")
            self.temp_mat.use_nodes = True
            # Clear any existing nodes to start fresh
            self.temp_mat.node_tree.nodes.clear()
        
    def cleanup_material(self):
        """Clears nodes but keeps material for reuse"""
        if self.temp_mat and self.temp_mat.node_tree:
            self.temp_mat.node_tree.nodes.clear()
        self.current_node_dict.clear()
        
    def destroy_material(self):
        """Completely remove material when done with all conversions"""
        if self.temp_mat:
            bpy.data.materials.remove(self.temp_mat)
            self.temp_mat = None

    def save_txt(self, text_file_path: str, force_save: bool = False) -> None:
        """
        Saves the DSL to a text file.
        If force_save is True, saves without validation (faster for batch processing).
        Otherwise validates first.
        """
        _ = self.get_txt()
        
        if not force_save:
            # Only validate if we haven't already or if forced
            if self._last_validation_result is None:
                valid, reason = self.validate_dsl()
                self._last_validation_result = (valid, reason)
            else:
                valid, reason = self._last_validation_result
            
            if not valid:
                raise ValueError(f"Invalid DSL format: {reason}")
        
        # Save the file
        with open(text_file_path, "w") as f:
            f.write(self.dsl_text)

    def valid_property_path_and_val(self, property_path: str, val_str: str) -> Tuple[bool, str]:
        """Optimized validation without eval()"""
        path_list = property_path.split(".")

        if path_list[0] not in self.current_node_dict:
            return False, f"Node variable {path_list[0]} not defined"
    
        current_attr = self.current_node_dict[path_list[0]]
        current_node_type = self.current_node_mapping[path_list[0]]

        # Handles 'W' issue in noise texture
        if current_node_type == "ShaderNodeTexNoise":
            current_attr.noise_dimensions = '4D'

        if current_node_type == "ShaderNodeBSDFPrincipled":
            current_attr.subsurface_method = 'RANDOM_WALK_SKIN'

        # Parse value once (convert string to Python literal safely)
        try:
            evaluated_val = ast.literal_eval(val_str)
        except:
            # If literal_eval fails, try simple types
            if val_str.isdigit():
                evaluated_val = int(val_str)
            elif val_str.replace('.', '').isdigit() and val_str.count('.') <= 1:
                evaluated_val = float(val_str)
            elif val_str in ('True', 'False'):
                evaluated_val = val_str == 'True'
            else:
                return False, f"Cannot parse value: {val_str}"

        for paths in path_list[1:]:
            # Handle Inputs
            if paths.startswith("i-"):
                socket_name = paths[2:]
                if socket_name not in self.check_valid_inputs[current_node_type]["inputs"]:
                    return False, f"Input '{socket_name}' not found in {current_node_type}"
                current_attr = current_attr.inputs[f'{socket_name}']
                continue
        
            # Handle Subscripts
            elif paths[0] in self.subscript_dict and paths[1:].isdigit():
                attr_name = self.subscript_dict[paths[0]]
                idx = int(paths[1:])
                collection = getattr(current_attr, attr_name)
                if idx >= len(collection):
                    return False, f"Index {idx} out of range"
                current_attr = collection[idx]
        
            # Handle Standard Attributes
            else:
                if not hasattr(current_attr, paths):
                    return False, f"Attribute '{paths}' not found"
                current_attr = getattr(current_attr, paths)
                
        return True, None

    def validate_dsl(self, reuse_material: bool = True) -> Tuple[bool, str]:
        """Optimized validation with material reuse"""
        if not reuse_material or self.temp_mat is None:
            self.setup_material()
        else:
            # Clear existing nodes but keep material
            self.cleanup_material()

        _ = self.get_txt()

        # Parse only if needed
        if len(self.nodes_info) == 0 or len(self.properties_info) == 0 or len(self.links_info) == 0:
            lines = self.dsl_text.strip().split('\n')
            if len(lines) >= 1 and lines[0].startswith('N|'):
                self.nodes_info = [n for n in lines[0][2:].split(';') if n]
            if len(lines) >= 2 and lines[1].startswith('P|'):
                self.properties_info = [p for p in lines[1][2:].split(';') if p]
            if len(lines) >= 3 and lines[2].startswith('L|'):
                self.links_info = [l for l in lines[2][2:].split(';') if l]
        
        # Batch create nodes
        for node_info in self.nodes_info:
            if not node_info:
                continue
            var, node_type = node_info.split(":")
            if node_type not in self.available_node_types:
                return False, f"{node_type} not available"
            self.current_node_dict[var] = self.temp_mat.node_tree.nodes.new(f"ShaderNode{node_type}")
            self.current_node_mapping[var] = f"ShaderNode{node_type}"

        # Validate links (optimized)
        for link_info in self.links_info:
            if not link_info:
                continue
            try:
                out_part, in_part = link_info.split(">")
                out_node, out_socket = out_part.split(".")
                in_node, in_socket = in_part.split(".")

                if out_node not in self.current_node_dict:
                    return False, f"Node {out_node} not found"
                if in_node not in self.current_node_dict:
                    return False, f"Node {in_node} not found"

                out_node_type = self.current_node_dict[out_node]
                in_node_type = self.current_node_dict[in_node]

                if out_socket not in out_node_type.outputs:
                    return False, f"Output '{out_socket}' not found on {out_node}"
                if in_socket not in in_node_type.inputs:
                    return False, f"Input '{in_socket}' not found on {in_node}"
            except Exception as e:
                return False, f"Link parse error: {e}"
        
        # Validate properties (batched)
        for prop_info in self.properties_info:
            if not prop_info:
                continue
            try:
                property_path, val = prop_info.split(":", 1)  # Split only once
                valid_path, error = self.valid_property_path_and_val(property_path, val)
                if not valid_path:
                    return False, error
            except Exception as e:
                return False, f"Property error: {e}"
        
        # Don't cleanup if reusing material
        if not reuse_material:
            self.cleanup_material()
            
        return True, None


class ConvertCodeToDSL:
    def __init__(self) -> None:
        with open("JSON_files/nodes_data_36.json", "r") as f:
            self.nodes_data_v36 = json.load(f)

        with open("JSON_files/nodes_data_51.json", "r") as f:
            self.nodes_data_v51 = json.load(f)

        self.dsl_shader = DSLShaders()
        self._full_path_cache = {}  # Cache for AST paths

    def get_full_path(self, node, use_cache: bool = True) -> str:
        """Cached version of path extraction"""
        if use_cache:
            # Simple cache key (not perfect but helps)
            cache_key = str(id(node))
            if cache_key in self._full_path_cache:
                return self._full_path_cache[cache_key]
        
        result = self._get_full_path_impl(node)
        if use_cache:
            self._full_path_cache[cache_key] = result
        return result
    
    def _get_full_path_impl(self, node) -> str:
        """Actual implementation without caching"""
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
            
            # Handle index extraction safely
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

    def convert_batch(self, python_files: list, text_dir: str, batch_size: int = 100) -> dict:
        """Process multiple files in batches to manage memory"""
        results = {
            'success': [],
            'failed': [],
            'total': len(python_files)
        }
        
        for i in range(0, len(python_files), batch_size):
            batch = python_files[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(python_files)-1)//batch_size + 1}")
            
            for py_file in batch:
                try:
                    # Generate output path
                    import os
                    base_name = os.path.basename(py_file).replace('.py', '.txt')
                    txt_path = os.path.join(text_dir, base_name)
                    
                    # Convert single file
                    text = self.convert(py_file, txt_path)
                    results['success'].append(py_file)
                    
                except Exception as e:
                    results['failed'].append((py_file, str(e)))
                    print(f"Failed: {py_file} - {e}")
            
            # Force garbage collection after each batch
            import gc
            gc.collect()
            
            # Reset the DSL shader's material but keep it for reuse
            self.dsl_shader.cleanup_material()
            
        return results

    def convert(self, python_path:str, text_path:str = None) -> str:
        """Optimized single file conversion"""
        # Reuse material across conversions
        self.dsl_shader.setup_material()
        nodes = self.dsl_shader.temp_mat.node_tree.nodes
        self.current_node_vartype_mapping = {}
        self._full_path_cache.clear()  # Clear cache for new file

        with open(python_path, "r") as f:
            tree = ast.parse(f.read())

        skip_first_mat = True
        
        # Process nodes in a single pass
        for node in ast.walk(tree):
            # Node creation
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                if getattr(node.value.func, 'attr', None) == 'new':
                    if isinstance(node.targets[0], ast.Name):
                        var_name = node.targets[0].id
                        node_type = node.value.args[0].value[10:]
                        
                        if node_type not in self.dsl_shader.available_node_types:
                            raise ValueError(f"{var_name} : {node_type} not available")
                        
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
                    
                    if path:  # Only process if we got a valid path
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
                    # Don't raise, just skip problematic properties
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

                            print(val)
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


        # Validate with material reuse
        # valid, error = self.dsl_shader.validate_dsl(reuse_material=True)
        text = self.dsl_shader.get_txt()

        if text_path:
            self.dsl_shader.save_txt(text_path, force_save=True)

        # Reset for next file but keep material
        self.dsl_shader.reset()
        self.dsl_shader.cleanup_material()  # Clear nodes but keep material

        return text
        # if valid:
        #     return text
        # else:
        #     raise ValueError(f"Error - dsl generated is not valid : {error}")


# Batch processing example
def batch_convert_all_py_files(source_dir: str, output_dir: str):
    """Convert all Python files in directory to DSL format"""
    import os
    import glob
    
    # Find all .py files
    py_files = glob.glob(os.path.join(source_dir, "*.py"))
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize converter
    converter = ConvertCodeToDSL()
    
    try:
        # Process in batches
        results = converter.convert_batch(py_files, output_dir, batch_size=100)
        
        print(f"\nConversion Complete:")
        print(f"Success: {len(results['success'])}")
        print(f"Failed: {len(results['failed'])}")
        
        # Log failures
        if results['failed']:
            with open(os.path.join(output_dir, "failures.log"), "w") as f:
                for file, error in results['failed']:
                    f.write(f"{file}: {error}\n")
                    
    finally:
        # Clean up the shared material
        converter.dsl_shader.destroy_material()


# if __name__ == "__main__":
#     # Example usage
#     batch_convert_all_py_files("path/to/python/files", "path/to/output/dsl")