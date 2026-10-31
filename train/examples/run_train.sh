#!/usr/bin/env bash
# sglang TP training example: 4 GPUs as one TP group, single training process.
# For the minimal single-GPU hf run see README.md.
# Adjust model path / data / schema before running.

set -e

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-0.6B}        # local dir or HF repo id
TRAIN_JSONL=${TRAIN_JSONL:-train.jsonl}
OUTPUT_PATH=${OUTPUT_PATH:-./output_example}
VERSION=${VERSION:-ema}

DATASET_INFO='{"task_a": {"label2id": {"normal": 0, "label_x": 1}}}'

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py \
    --model_name_or_path="${MODEL_PATH}" \
    --version="${VERSION}" \
    --input_table="${TRAIN_JSONL}" \
    --output_path="${OUTPUT_PATH}" \
    --feature_layers=0,1,2,3,4 \
    --selected_layers=5,10,15,20,25 \
    --llm_backend=sglang --tp_size=4 --gpu_memory_utilization=0.6 \
    --dataset_info="${DATASET_INFO}" \
    --max_length=8192 --chunked_prefill_size=8192 \
    --batch_size=2 --accumulation_steps=16 --epochs=2 --learning_rate=5e-5 \
    --text_col=messages --label_col=label --overlap_mode=front \
    --marker_max_id=25 --marker_sight=1,10
