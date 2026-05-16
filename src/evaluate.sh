#!/bin/bash
# evaluate.sh

DATA_PATH="lora_0.1_eval/train_eval"
MODEL_PATH="model_ckpts/run_0.1_lora"
JSON_PATH="lora_0.1_eval/train_response_run_0.1_response_01.json"
ENDPOINT="https://ffc95aec1330d2d20a.gradio.live/"

# usage() {
#     echo "Usage: $0 [OPTIONS]"
#     echo ""
#     echo "Options:"
#     echo "  --data-path       PATH    Input data path         (default: $DATA_PATH)"
#     echo "  --model-path      PATH    Model ckpt path         (default: $MODEL_PATH)"
#     echo "  --json-path       PATH    Inference output JSON   (default: $JSON_PATH)"
#     echo "  --endpoint        PATH    gradio endpoint         (default: $ENDPOINT)"
#     echo "  -h, --help                Show this help message"
#     exit 1
# }

# while [[ $# -gt 0 ]]; do
#     case "$1" in
#         --data-path)
#             DATA_PATH="$2"
#             shift 2
#             ;;
#         --model-path)
#             MODEL_PATH="$2"
#             shift 2
#             ;;
#         --json-path)
#             JSON_PATH="$2"
#             shift 2
#             ;;
#         --endpoint)
#             ENDPOINT="$2"
#             shift 2
#             ;;
#         -h|--help)
#             usage
#             ;;
#         *)
#             echo "Unknown argument: $1"
#             usage
#             ;;
#     esac
# done

# echo "--- Config ---"
# echo "  Data path:      $DATA_PATH"
# echo "  Model path:     $MODEL_PATH"
# echo "  JSON path:      $JSON_PATH"
# echo "  ENDPOINT:       $ENDPOINT"
# echo ""

echo "---------- Inference ----------"
python -m src.model.evaluate.infer \
    --data_path "$DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --save_json_path "$JSON_PATH" \
    --gradio_endpoint "$ENDPOINT"

echo "---------- Rendering ----------"
/mnt/Storage/ML/blender-5.1.0-linux-x64/blender \
    --background \
    --python src/model/evaluate/blender_render.py \
    -- \
    --save_json_path "$JSON_PATH"

echo "---------- Scoring ----------"
python -m src.model.evaluate.similarity_score \
    --save_json_path "$JSON_PATH"

echo "--- Done ---"