from transformers import AutoProcessor
import json
import argparse
from tqdm.auto import tqdm
from pathlib import Path 

def get_new_tokens(
        dataset_path: str,
        vocab: dict
) -> list[str]:
    
    txt_files = Path(dataset_path).rglob("*.txt")

    check_tokens = set()

    for txt_file in tqdm(txt_files):
        try:
            if txt_file.name == "blender_full.txt":
                continue

            with open(txt_file, "r") as f:
                shader_text = f.read()

            lines = shader_text.split("\n")

            for line in lines:
                line.strip()
            
                if line.startswith("N|"):
                    nodes = line[2:].split(";")
                    if not nodes or nodes == ['']:
                        continue
                    for node_info in nodes:
                        var_name, type_name = node_info.split(":")
                        if var_name not in check_tokens and var_name not in vocab:
                            check_tokens.add(var_name)
                        if type_name not in check_tokens and type_name not in vocab:
                            check_tokens.add(type_name)

                if line.startswith("P|"):
                    props = line[2:].split(";")
                    if not props or props == ['']:
                        continue
                    for prop_info in props:
                        prop_path, value = prop_info.split(":")
                        prop_names = prop_path.split(".")
                        for name in prop_names:
                            if name.startswith("i-"):
                                name = name[2:]
                            if name not in check_tokens and name not in vocab:
                                check_tokens.add(name)

                if line.startswith("L|"):
                    links = line[2:].split(";")
                    if not links or links == ['']:
                        continue
                    for link_info in links:
                        out_node, in_node = link_info.split(">")
                        link_names = out_node.split(".") + in_node.split(".")
                        for name in link_names:
                            if name not in check_tokens and name not in vocab:
                                check_tokens.add(name)
        except Exception as e:
            raise RuntimeError(f"file - {txt_file} got error - {e}")

    new_tokens = list(check_tokens)
    return new_tokens

def main(
        dataset_path: str,
        save_json_path: str
) -> None:
    # initialize processor & get vocab
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B")
    vocab = processor.tokenizer.get_vocab()

    # get the new_tokens
    new_tokens = get_new_tokens(dataset_path=dataset_path, vocab=vocab)
    special_tokens = ["N|", "P|", "L|", "i-", ".dv"]

    # export the new tokens & special tokens in a json file
    with open(save_json_path, "w") as f:
        json.dump({
            "new_tokens" : new_tokens,
            "special_tokens" : special_tokens
        }, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        required=True
    )
    parser.add_argument(
        "--save_json_path",
        required=True
    )
    args = parser.parse_args()
    main(
        dataset_path=args.data_path,
        save_json_path=args.save_json_path
    )

"""
python -m src.data.additional_tokens \
--data_path ShaderDataset \
--save_json_path JSON_files/addition_tokens.json
"""