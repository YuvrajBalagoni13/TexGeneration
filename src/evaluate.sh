#!/bin/bash
set -e

MODEL_BASE=""
LORA_PATH=""
EVAL_DATA_PATH=""
OUTPUT_JSON_PATH=""
RESULT_JSON_PATH=""
RENDER_PATH=""
DATA_LENGTH=100
BATCH_SIZE=4
BATCH_PROCESS=false
NEW_TOKENS=false
QUANTIZE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_base)
            MODEL_BASE="$2"
            shift 2;;
        --lora_path)
            LORA_PATH="$2"
            shift 2;;
        --eval_data_path)
            EVAL_DATA_PATH="$2"
            shift 2;;
        --output_json_path)
            OUTPUT_JSON_PATH="$2"
            shift 2;;
        --result_json_path)
            RESULT_JSON_PATH="$2"
            shift 2;;
        --render_path)
            RENDER_PATH="$2"
            shift 2;;
        --data_length)
            DATA_LENGTH="$2"
            shift 2;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2;;
        --quantize)
            QUANTIZE=true
            shift;;
        --batch_process)
            BATCH_PROCESS=true
            shift;;          
        --new_tokens)
            NEW_TOKENS=true
            shift;;
        *)
            echo "Unknown argument: $1"
            exit 1;;
    esac
done

BATCH_FLAG=""
if [ "$BATCH_PROCESS" = true ]; then
    BATCH_FLAG="--batch_process"
fi

QUANTIZE_FLAG=""
if [ "$QUANTIZE" = true ]; then
    QUANTIZE_FLAG="--quantize"
fi 

NEW_TOKENS_FLAG=""
if [ "$NEW_TOKENS" = true ]; then
    NEW_TOKENS_FLAG="--new_tokens"
fi

python -m src.model.infer \
--model_base "$MODEL_BASE" \
--lora_path "$LORA_PATH" \
--eval_data_path "$EVAL_DATA_PATH" \
--save_json_path "$OUTPUT_JSON_PATH" \
--data_length "$DATA_LENGTH" \
--batch_size "$BATCH_SIZE \
$QUANTIZE_FLAG \
$BATCH_FLAG \
$NEW_TOKEN_FLAG

/mnt/Storage/ML/blender-5.1.0-linux-x64/blender \
--background \
--python src/model/renderer.py \
-- \
--result_json_path "$OUTPUT_JSON_PATH" \
--save_json_path "$RESULT_JSON_PATH" \
--render_path "$RENDER_PATH"

python -m src.model.similarity_score \
--result_json_path "$RESULT_JSON_PATH"