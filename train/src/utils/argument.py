from typing import Optional
from dataclasses import dataclass, field


@dataclass
class TrainArguments:
    job_name: str = field(
        default=None,
        metadata={"help": "Job name, recorded in the saved config only."}
    )
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "The model path of pretrained model."}
    )
    output_path: str= field(
        default=None,
        metadata={"help": "Directory for storing models."}
    )
    dataset_dir: str= field(
        default=None,
        metadata={"help": "Directory of training dataset."}
    )
    model_name: Optional[str] = field(
        default="PlugGuard",
        metadata={"help": "Exported model name."}
    )
    version: Optional[str] = field(
        default=None,
        metadata={"help": "Model version, one of: lasttok / ema / gru."}
    )
    max_length: Optional[int] = field(
        default=512,
        metadata={"help": "The max length of input sequence"}
    )
    batch_size: Optional[int] = field(
        default=128,
        metadata={"help": "The batch size in training."}
    )
    epochs: Optional[int] = field(
        default=1,
        metadata={"help": "Number of training epochs."}
    )
    learning_rate: Optional[float] = field(
        default=2e-5,
        metadata={"help": "The learning rate used for training."}
    )
    dropout_prob: Optional[float] = field(
        default=0.1,
        metadata={"help": "Dropout rate."}
    )
    dropout_type: Optional[str] = field(
        default="stable",
        metadata={"help": "Dropout type."}
    )
    overwrite: Optional[bool] = field(
        default=True,
        metadata={"help": "Allow writing into an existing output directory."}
    )

    dataset_info: Optional[str] = field(
        default=None,
        metadata={"help": "Inline dataset_info JSON, used when dataset_dir is not provided."}
    )
    input_table: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the local jsonl training data file."}
    )
    input_table_worker: Optional[int] = field(
        default=8,
        metadata={"help": "DataLoader num_workers for reading the training data."}
    )
    label_type: Optional[str] = field(
        default='single',
        metadata={"help": "Label type; only 'single' (single-label) is supported."},
    )
    use_wrap_model: Optional[bool] = field(
        default=True,
        metadata={"help": "Use the wrapped model (feature extractor + classification heads)."}
    )
    feature_layers: Optional[str] = field(
        default=None,
        metadata={"help": '\"all\" (layers 1..N-1) or comma-separated indices into captured representations; required for training, with at least three layers for current heads.'}
    )
    accumulation_steps: Optional[int] = field(
        default=1,
        metadata={"help": "Number of gradient accumulation steps."}
    )

    text_col: Optional[str] = field(
        default='text',
        metadata={"help": "Text column of the training data."}
    )
    label_col: Optional[str] = field(
        default='label',
        metadata={"help": "Label column of the training data."}
    )
    image_col: Optional[str] = field(
        default=None,
        metadata={"help": "Image column of the training data (VL models only; leave empty for text-only models)."}
    )
    max_pixels: Optional[int] = field(
        default=448*448,
    )
    min_pixels: Optional[int] = field(
        default=224*224,
    )

    marker_pattern: Optional[str] = field(
        default=r"<MARKER###(\d+)>",
        metadata={"help": "Marker regex with one capture group containing an integer id."}
    )
    marker_max_id: Optional[int] = field(
        default=1,
        metadata={"help": "marker id range is [1, max_id]"}
    )
    marker_sight: Optional[str] = field(
        default='2,512,1,10',
        metadata={"help": "Left-context mask as alternating value,length pairs; later regions overwrite earlier ones, and every value >0 contributes to loss."}
    )
    overlap_mode: Optional[str] = field(
        default='front',
        metadata={"help": "How overlapping marker masks are resolved: 'none' (allow overlap) / 'front' (smaller id wins) / 'back' (larger id wins)", "choices": ["none", "front", "back"]},
    )

    llm_backend: Optional[str] = field(
        default="hf",
        metadata={"help": "Backbone backend: single-GPU HF, multi-GPU HF device_map=auto (hfpp), or SGLang.", "choices": ["hf", "sglang", "hfpp"]},
    )

    tp_size: Optional[int] = field(
        default=1,
    )
    selected_layers: Optional[str] = field(
        default="all",
        metadata={"help": "Layers extracted by sglang, e.g. 'all' or '0,10,20,28'"}
    )
    chunked_prefill_size: Optional[int] = field(
        default=None,
        metadata={"help": "sglang chunked prefill token budget; -1 disables it (recommended), >0 is the max tokens per prefill"}
    )
    max_prefill_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "sglang max tokens per prefill batch (default 16384); raise it to cover the full batch token count for large batches"}
    )
    gpu_memory_utilization: Optional[float] = field(
        default=None,
    )
    attention_backend: Optional[str] = field(
        default=None,
        metadata={"help": "SGLang attention backend; None lets SGLang choose. On AMD, triton_noaiter disables aiter for vLLM compatibility; aiter_lowmem reduces loading memory."}
    )
