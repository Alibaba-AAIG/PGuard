# python3 -m pguard_vllm.heads : registry + real-checkpoint self-test.
#
# The registry self-test lives here (not in __init__.py) so the package can
# be executed with -m; the assertions run against the SAME registered classes
# the engine resolves at load time.

import os

import torch

from . import build_head_from_checkpoint
from .base import GuardHeadRegistry


def main() -> int:
    assert GuardHeadRegistry.kinds() == ["ema", "gru", "lasttok"], \
        GuardHeadRegistry.kinds()
    for kind in ("lasttok", "ema", "gru"):
        assert GuardHeadRegistry.resolve(kind).kind == kind

    # real checkpoints (if present): strict state-dict load + vendored forward
    pw = (os.environ.get("PGUARD_PW_DIR") or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..",
                     "pg_weight")))
    name_map = {"lasttok": "Lasttok", "ema": "EMA", "gru": "GRU"}
    ran = 0
    for variant in ("lasttok", "ema", "gru"):
        d = os.path.join(pw, f"PGuard-{name_map[variant]}-Qwen3.5-9B")
        if not os.path.isdir(d):
            print(f"skip {variant}: {d} not found")
            continue
        cfg, head = build_head_from_checkpoint(d)
        assert cfg.pguard_version == variant
        h = head.to(torch.float32)
        ones = torch.ones(2, 5, dtype=torch.bool)
        feats = torch.randn(2, 3, 5, cfg.hidden_size)
        with torch.no_grad():
            if variant == "gru":
                o3 = h.forward_prefill(feats, mask=ones)
                n = len(o3[0])
            else:
                o3 = h(features=feats, token_type_ids=None, label_type=None,
                       label_marker=None, attention_mask=ones)
                n = len(o3.full_classification_logits)
        assert n == len(cfg.classifier["classification_heads"])
        ran += 1
        print(f"{variant} checkpoint forward OK ({n} heads)")
    print("heads self-test OK (registry:", GuardHeadRegistry.kinds(),
          "; checkpoints ran:", ran, ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
