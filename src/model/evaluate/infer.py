import json
from pathlib import Path
from tqdm.auto import tqdm
from gradio_client import Client, handle_file
import argparse
import httpx

def load_eval_data(eval_data_path: str = "") -> list[dict]:
    image_path_list = list(Path(eval_data_path).rglob("*.jpg"))
    data = []
    for image_path in image_path_list:
        sample = {}
        sample["image"] = str(image_path)
        with open(image_path.with_suffix(".txt"), "r") as f:
            sample["shader"] = f.read()
        data.append(sample)
    print(f"---------- Total eval data length = {len(data)} samples ----------")
    return data

def main(
        data_path: str = "",
        model_path: str = "",
        save_response_path: str = "",
        gradio_endpoint: str = "https://84836944933253a238.gradio.live/"
        ) -> None:
    # load eval data
    eval_data = load_eval_data(eval_data_path=data_path)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    client = Client(
        gradio_endpoint,
        httpx_kwargs={"timeout": httpx.Timeout(6000.0)}
        )

    input_prompt = ("Generate a text based shader graph in the following format -\n"
                    "N|node_name:node_type;...\n"
                    "P|node_name.property_path:value;...\n"
                    "L|node_name.output_socket>node_name.input_socket;...\n"
                    "Here N| represents nodes, P| tells properties & L| tells links.")
    
    responses = {}

    for i, sample in tqdm(enumerate(eval_data)):
        sample_report = {}
        sample_report["image_path"] = str(sample["image"])
        sample_report["actual_shader"] = sample["shader"]

        output_txt_shader = client.predict(
        	image=handle_file(str(sample["image"])),
        	prompt=input_prompt,
        	api_name="/handle_inference"
        )

        sample_report["output_shader"] = output_txt_shader
        responses[f"sample_{i}"] = sample_report

    with open(save_response_path, "w") as f:
        json.dump(responses, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        required=True,
        default="Dataset/eval/train_eval"
    )

    parser.add_argument(
        "--model_path",
        required=True,
        default="model_ckpts/run_0.1_lora"
    )

    parser.add_argument(
        "--save_json_path",
        required=True,
        default="Dataset/eval/train_response_run_0.1_response.json"
    )

    parser.add_argument(
        "--gradio_endpoint",
        required=True,
        default="https://84836944933253a238.gradio.live/"
    )

    args = parser.parse_args()

    main(
        data_path=args.data_path,
        model_path=args.model_path,
        save_response_path=args.save_json_path,
        gradio_endpoint=args.gradio_endpoint
        )
    
"""
python -m src.model.evaluate.infer \
    --data_path "Dataset/eval/train_eval" \
    --model_path "model_ckpts/run_0.1_lora" \
    --save_json_path "Dataset/eval/train_response_run_0.1_response_01.json" \
    --gradio_endpoint "https://84836944933253a238.gradio.live/"
"""