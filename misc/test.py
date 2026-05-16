import ast 

def get_full_path(node) -> str:
        if isinstance(node, ast.Name):
            print(node.id)
            return node.id
        elif isinstance(node, ast.Attribute):
            base = get_full_path(node.value)
            # if node.attr == "default_value":
            #     return base
            print(f"{base}.{node.attr}")
            return f"{base}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            print(node.value)
            base = get_full_path(node.value)
            base_name_list = base.split(".")
            base_name_list[-1] = base_name_list[-1][0]
            base = ".".join(base_name_list)
            index = node.slice.value if isinstance(node.slice, ast.Constant) else "x"

            if base_name_list[-1][0] == "i":
                 pass
                # node_type = self.dsl_shader.current_node_dict[base[0]]
                # attr_name = self.nodes_data_v36[node_type]["inputs"][index]
                # if attr_name not in self.nodes_data_v51[node_type]["inputs"]:
                #     raise ValueError(f"{attr_name} not as input to {node_type}.")
                # return f"{base}-{attr_name}"
            print(f"{base}{index}")
            return f"{base}{index}"
        return ""

with open("material_dataset_filtered/B3DMatPack1.2/AB_cendre/var_00000_full.py", "r") as f:
    tree = ast.parse(f.read())

for node in ast.walk(tree):
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Attribute):
        target = node.targets[0]
        path = get_full_path(target)
        print(path)