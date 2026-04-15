from tqdm.auto import tqdm
import os
import json
from pathlib import Path
import ast
from collections import defaultdict

def get_full_path(node):
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        base = get_full_path(node.value)
        if node.attr == "default_value":
            return base
        return f"{base}.{node.attr}"
    elif isinstance(node, ast.Subscript):
        base = get_full_path(node.value)
        base_name_list = base.split(".")
        base_name_list[-1] = base_name_list[-1][0]
        base = ".".join(base_name_list)
        index = node.slice.value if isinstance(node.slice, ast.Constant) else "x"
        return f"{base}{index}"
    return ""

def get_path_from_name(name, data_path):
    splitted_name = name.split("-")
    current_path = data_path
    for name in splitted_name[:-1]:
        current_path = current_path / name 
        current_path.mkdir(parents=True, exist_ok=True)
    current_path = current_path / splitted_name[-1]
    return current_path

def get_json_from_py(file_path, nodes_count=None):
    with open(file_path, "r") as f:
        tree = ast.parse(f.read())
    
    nodes, props, links = [], [], []
    start = True
    for node in ast.walk(tree):
    
        # for shader nodes selection
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if getattr(node.value.func, 'attr', None) == 'new':
                var_name = node.targets[0].id
                node_type = node.value.args[0].value[10:]
                nodes.append(f"{var_name}:{node_type}")
                if nodes_count != None:
                    nodes_count[node_type] += 1

        # for properties   
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Attribute):
            if start: 
                start = False
                continue
            target = node.targets[0]
            path = get_full_path(target)
            try:
                # literal_eval handles numbers, strings, and lists [r,g,b,a]
                val = ast.literal_eval(node.value)
               
                if isinstance(val, str): val = f"'{val}'"
                if isinstance(val, float): val = round(val, 3)
                if isinstance(val, list): val = [round(x, 2) if isinstance(x, float) else x for x in val]  

                props.append(f"{path}:{val}")
            except:
                pass

        # for links
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if getattr(node.value.func, 'attr', None) == 'new':
                out_part = node.value.args[0]

                # handling any .new attribute of nodes.
                if not isinstance(out_part, ast.Subscript):
                    target = node.value.func
                    path = get_full_path(target)
                    try:
                        if len(node.value.args) != 1:
                            val = []
                            for arg in node.value.args:
                                val.append(arg.value)
                            if isinstance(val, str): val = f"'{val}'"
                            if isinstance(val, float): val = round(val, 3)
                            if isinstance(val, list): val = [round(x, 2) if isinstance(x, float) else x for x in val]
                        else:
                            val = node.value.args[0].value
                            if isinstance(val, str): val = f"'{val}'"
                            if isinstance(val, float): val = round(val, 3)
                            if isinstance(val, list): val = [round(x, 2) if isinstance(x, float) else x for x in val]

                        props.append(f"{path}:{val}")
                    except Exception as e:
                        print(f"Was not able to process {path} because {e}")
                    continue
                
                in_part = node.value.args[1]
                links.append(f"{out_part.value.value.id}.{out_part.slice.value}>{in_part.value.value.id}.{in_part.slice.value}")

    return f"N|{';'.join(nodes)}\nP|{';'.join(props)}\nL|{';'.join(links)}"


def main(dataset_dir="material_dataset_filtered"):
    nodes_count = defaultdict(int)

    data_save_path = Path("CurrentDataset")
    txt_data_save_path = data_save_path / "txt"

    dataset_path = Path(dataset_dir)
    print("Starting conversion...")

    with open("groups.json", "r") as f:
        groups_dict = json.load(f)
    
    group_set = set()
    for grp in groups_dict["groups"]:
        if grp in group_set: continue 
        else: group_set.add(grp)

    py_files = list(dataset_path.rglob("*.py"))

    for file_path in tqdm(py_files, desc="Converting files"):
        if str(file_path) in group_set:
            continue

        txt_file_name = "-".join(str(file_path.with_suffix(".txt")).split("/")[1:])
        txt_file_path = get_path_from_name(txt_file_name, txt_data_save_path)

        # if txt_file_path.exists():
        #     continue

        try:
            text = get_json_from_py(file_path, nodes_count)
            
            with open(txt_file_path, "w", encoding="utf-8") as f:
                f.write(text)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print("Completed conversion.")

    with open("node_count.json", "w") as f:
        json.dump(nodes_count, f, indent=4)

if __name__ == "__main__":
    main("material_dataset_filtered")


    # text = get_json_from_py("material_dataset_filtered/mat_llm_r2/case_01048_gen_03/var_00009_full.py")
    # with open("test_file.txt", "w") as f:
    #     f.write(text)
    #     print(text)