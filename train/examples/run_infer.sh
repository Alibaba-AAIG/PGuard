#!/usr/bin/env bash
# sglang TP inference example with extended per-role output.
# For the minimal single-GPU hf run see README.md.
# Adjust model paths / data before running.

set -e

MODEL_PATH=${MODEL_PATH:-./output_example}      # trained head dir or HF repo id
INPUT_JSONL=${INPUT_JSONL:-input.jsonl}
OUTPUT_JSONL=${OUTPUT_JSONL:-result.jsonl}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

python infer.py \
    --model_name_or_path="${MODEL_PATH}" \
    --input_table="${INPUT_JSONL}" \
    --output_table="${OUTPUT_JSONL}" \
    --batch_size=1 \
    --max_length=8192 --chunked_prefill_size=8192 \
    --llm_backend=sglang --tp_size=2 --gpu_memory_utilization=0.8 \
    --feature_layers=0,1,2,3,4 \
    --selected_layers=5,10,15,20,25 \
    --input_col=input_text \
    --return_extend
