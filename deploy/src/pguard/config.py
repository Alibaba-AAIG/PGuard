import os


def is_debug() -> bool:
    return os.getenv("PGUARD_DEBUG", "") == "1"


# ======================================================================
# ======================================================================
PGUARD_MODELING_VARIANT = os.getenv("PGUARD_MODELING_VARIANT")


def modeling_variant() -> str:
    v = (PGUARD_MODELING_VARIANT or "").strip().lower()
    if not v:
        raise ValueError(
            "PGUARD_MODELING_VARIANT is required (e.g. 'stateless' / 'seq' / 'cfc'), but not set"
        )
    return v


PGUARD_OVERWRITE_MODEL = os.getenv("PGUARD_OVERWRITE_MODEL", "") == "1"
PGUARD_CAPTURE_LAYERS = os.getenv("PGUARD_CAPTURE_LAYERS", "")


def capture_layer_indices():
    return [int(x) for x in PGUARD_CAPTURE_LAYERS.split(",") if x.strip()]


PGUARD_SKIP_FIRST_TOKEN = bool(int(os.getenv("PGUARD_SKIP_FIRST_TOKEN", 0)))
PGUARD_COPY_LOGITS_PROB = float(os.getenv("PGUARD_COPY_LOGITS_PROB", 0.1))


def _score_interval(prob: float) -> int:
    if prob <= 0.0:
        return 0
    if prob >= 1.0:
        return 1
    return max(1, round(1.0 / prob))


PGUARD_SCORE_INTERVAL = _score_interval(PGUARD_COPY_LOGITS_PROB)
PGUARD_ASYNC_COPY = os.getenv("PGUARD_ASYNC_COPY", "0") == "1"
#         patches/guard_variant.py + patches/breakable_patch.py
PGUARD_BREAKABLE_GRAPH = os.getenv("PGUARD_BREAKABLE_GRAPH", "0") == "1"

# ======================================================================
# （sglang/variants/qwen3_5_pguard_stateless.py + guard/plugin_guard.py + guard/guard_utils.py）
# ======================================================================
PGUARD_MODEL_PATH = os.getenv("PGUARD_MODEL_PATH")
PGUARD_FUSE_MERGE = bool(int(os.getenv("PGUARD_FUSE_MERGE", 1)))
PGUARD_FUSE_ADAPTER = bool(int(os.getenv("PGUARD_FUSE_ADAPTER", 1)))
PGUARD_FUSE_CLASSIFIERS = bool(int(os.getenv("PGUARD_FUSE_CLASSIFIERS", 1)))
PGUARD_GATE_TEMPERATURE = float(os.getenv("PGUARD_GATE_TEMPERATURE", 1.0))

# ======================================================================
# （sglang/variants/qwen3_5_pguard_seq.py + guard/seq_guard.py + guard/history_state.py）
# ======================================================================
PGUARD_HIST_N = int(os.getenv("PGUARD_HIST_N", "0"))
PGUARD_EMA_SCALES = tuple(
    float(x) for x in os.getenv("PGUARD_EMA_SCALES", "200,150,100,50").split(",") if x.strip()
)
PGUARD_PREFILL_RISK_MODE = os.getenv("PGUARD_PREFILL_RISK_MODE", "last").strip().lower()
PGUARD_PREFILL_SCORE_PROB = float(os.getenv("PGUARD_PREFILL_SCORE_PROB", "1.0"))
PGUARD_SEQ_FUSE_KERNELS = bool(int(os.getenv("PGUARD_SEQ_FUSE_KERNELS", "0")))
PGUARD_SEQ_FUSE_EMA = bool(int(os.getenv("PGUARD_SEQ_FUSE_EMA", "0")))
PGUARD_SEQ_FUSE_CLASSIFIERS = bool(
    int(os.getenv("PGUARD_SEQ_FUSE_CLASSIFIERS", "0"))
)
PGUARD_SEQ_DECODE_TP_FUSION = bool(
    int(os.getenv("PGUARD_SEQ_DECODE_TP_FUSION", "0"))
)
PGUARD_SEQ_COMPILE = bool(int(os.getenv("PGUARD_SEQ_COMPILE", "0")))
PGUARD_SEQ_SAMPLED_SCAN_CHUNK = int(
    os.getenv("PGUARD_SEQ_SAMPLED_SCAN_CHUNK", "128")
)
PGUARD_SEQ_FUSED_SAMPLED_EMA = bool(
    int(os.getenv("PGUARD_SEQ_FUSED_SAMPLED_EMA", "0"))
)
PGUARD_SEQ_PREFILL_OVERLAP = bool(
    int(os.getenv("PGUARD_SEQ_PREFILL_OVERLAP", "0"))
)
PGUARD_VERSION = os.getenv("PGUARD_VERSION", "").strip()
PGUARD_SEQ_NUM_CLASSES = int(os.getenv("PGUARD_SEQ_NUM_CLASSES", "4"))
PGUARD_SEQ_STATE_PATH = os.getenv("PGUARD_SEQ_STATE_PATH", "").strip()
PGUARD_MAX_REQ = int(os.getenv("PGUARD_MAX_REQ", "256"))
PGUARD_SCORE_KMAX_SIGMA = float(os.getenv("PGUARD_SCORE_KMAX_SIGMA", "4"))
PGUARD_SEQ_ROW_SHARD = bool(int(os.getenv("PGUARD_SEQ_ROW_SHARD", "0")))

# ======================================================================
# ======================================================================
PGUARD_CFC_PROJ_DIM = int(os.getenv("PGUARD_CFC_PROJ_DIM", "768"))
PGUARD_CFC_MEM_DIM = int(os.getenv("PGUARD_CFC_MEM_DIM", "768"))
PGUARD_CFC_USE_DT = bool(int(os.getenv("PGUARD_CFC_USE_DT", "0")))
PGUARD_CFC_DT = float(os.getenv("PGUARD_CFC_DT", "1.0"))
PGUARD_CFC_NUM_CLASSES = int(os.getenv("PGUARD_CFC_NUM_CLASSES", "4"))
PGUARD_CFC_STATE_PATH = os.getenv("PGUARD_CFC_STATE_PATH", "").strip()
PGUARD_CFC_GRAPH_UNROLL = bool(int(os.getenv("PGUARD_CFC_GRAPH_UNROLL", "1")))
PGUARD_CFC_USE_TRITON = bool(int(os.getenv("PGUARD_CFC_USE_TRITON", "0")))
PGUARD_CFC_TRITON_GRU = bool(int(os.getenv("PGUARD_CFC_TRITON_GRU", "1")))
PGUARD_CFC_LOOP_UNPACK = bool(int(os.getenv("PGUARD_CFC_LOOP_UNPACK", "0")))
PGUARD_CFC_ENABLE_FUSION = bool(int(os.getenv("PGUARD_CFC_ENABLE_FUSION", "0")))
PGUARD_CFC_FUSE_ADAPTER_MIN_ROWS = int(os.getenv("PGUARD_CFC_FUSE_ADAPTER_MIN_ROWS", "64"))
PGUARD_CFC_PREFILL_TP_DEDUP = bool(int(os.getenv("PGUARD_CFC_PREFILL_TP_DEDUP", "0")))
PGUARD_CFC_PREFILL_MODE = os.getenv("PGUARD_CFC_PREFILL_MODE", "pool").strip().lower()
PGUARD_CFC_PREFILL_EFFECTIVE_LENGTHS = tuple(
    int(x) for x in os.getenv("PGUARD_CFC_PREFILL_EFFECTIVE_LENGTHS", "200,150,100,50").split(",")
    if x.strip()
)
PGUARD_CFC_COMPILE = bool(int(os.getenv("PGUARD_CFC_COMPILE", "0")))
PGUARD_CFC_STACK_BUFFER = bool(int(os.getenv("PGUARD_CFC_STACK_BUFFER", "0")))
PGUARD_CFC_HF_ALIGNED_BACKBONE = bool(int(os.getenv("PGUARD_CFC_HF_ALIGNED_BACKBONE", "0")))
