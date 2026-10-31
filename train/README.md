# PlugGuard

PlugGuard is a plug-in guardrail built on a frozen foundation model backbone.

```
messages -> frozen LLM -> per-layer hidden states -> trainable head -> per-task safety labels
```

## Supported models

| Model family | Type |
|---|---|
| Qwen3 | text |
| Qwen3-VL | image-text |
| Qwen3.5 | image-text |
| GLM-4.5V | image-text |
| DeepSeek-V4-Flash | text |

## Installation

```bash
pip install -r requirements.txt
pip install -r requirements_dsv4.txt    # DeepSeek-V4 (sglang 0.5.13, AMD ROCm)
```

## Data format

Training and inference both read local JSONL files.

**Training**:

```json
{"messages": [{"role": "user", "content": "some text<MARKER###1>more text<MARKER###2>"}],
 "label": {"1": ["task_a###label_x"], "2": ["task_a###label_y", "task_b###label_z"]}}
```

- Markers must already be present in the training text. Each marker id is a key of `label`, mapping to a list of `task###label` strings.
- Tasks and their `label2id` are declared via `--dataset_info`. Each task becomes one classification head. Every marker carries exactly one label per head.

**Inference input**:

```json
{"traceid": "t1", "input_text": [{"role": "user", "content": "text to score"}]}
```

**Inference output**: 

```json
{"traceid": "t1", "result": "{\"task_a\": {\"label_x\": [0.31, 0.3, 0.28, 0.42, 0.45, 0.44, 0.41, 0.4, 0.35, 0.33, 0.34, 0.36], \"normal\": [0.68, 0.69, 0.71, 0.57, 0.54, 0.55, 0.58, 0.59, 0.64, 0.66, 0.65, 0.63]}}"}
```

## Minimal training and inference run

Train the `ema` head on a small Qwen3 model with the `hf` backend:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --version ema \
    --input_table train.jsonl \
    --output_path ./output \
    --dataset_info '{"task_a": {"label2id": {"normal": 0, "label_x": 1}}}' \
    --feature_layers 0,1,2,3,4,5,6,7,8 \
    --llm_backend hf \
    --max_length 8192 --batch_size 2 --epochs 2 --learning_rate 5e-5 \
    --text_col messages --label_col label --overlap_mode front \
    --marker_max_id 25 --marker_sight 1,10
```

Score new conversations with the trained head:

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --model_name_or_path ./output \
    --input_table input.jsonl \
    --output_table result.jsonl \
    --llm_backend hf \
    --max_length 8192 --batch_size 1 \
    --input_col input_text
```

Multi-GPU and sglang-backend variants are in `examples/`