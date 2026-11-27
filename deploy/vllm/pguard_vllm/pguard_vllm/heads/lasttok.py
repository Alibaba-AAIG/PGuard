# Vendored PGuard Lasttok head ("PluginGuard", pguard_version=lasttok), from
# Alibaba-AAIG/PGuard train/src/model/modeling_pguard_lasttok.py (Apache-2.0,
# inference path only) + the engine-facing score_step contract.
#
# Stateless: every scored token is an independent classification over the
# stacked capture-layer features. Prefill breadth follows
# PGUARD_PREFILL_RISK_MODE (`all` = every prompt token, enables
# prompt_series; `last` = each request's final prompt token only, the
# official default; decode rows are n=1 so `last` is a no-op for them).
#
# In-module self-test: python3 -m pguard_vllm.heads.lasttok

import torch
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertConfig

from ..pguard_config import get_config, reload_config
from .base import GuardHeadBase, GuardHeadRegistry, HeadStepResult, pad_rows
from .common import (AdapterBlock, BasicClassificationHead, config_from_dict,
                     classifiers_forward, fused_adapter_forward,
                     PGuardMultiTasksOutputForCls)


@GuardHeadRegistry.register("lasttok")
class PluginGuard(GuardHeadBase):
    """Consumes stacked multi-layer hidden states (bs, layer, len, hidden)."""

    def __init__(self, config):
        super().__init__(config)
        self.top_L = 3
        self.gate = torch.nn.Linear(config.hidden_size, 1)
        self.gate_temperature = 1.0

        bert_layer_config = BertConfig()
        bert_layer_config.add_cross_attention = False

        self.input_proj = torch.nn.Linear(config.hidden_size,
                                          bert_layer_config.hidden_size)
        self.adapter = torch.nn.ModuleList(
            [AdapterBlock(bert_layer_config) for _ in range(3)]
        )
        self.norm = torch.nn.LayerNorm(config.hidden_size,
                                       eps=bert_layer_config.layer_norm_eps)

        self.classifiers = torch.nn.ModuleList()
        for name, num_classes in config.classifier["classification_heads"]:
            classifier_config = config_from_dict({
                "hidden_size": bert_layer_config.hidden_size,
                "dropout_prob": config.classifier["dropout_prob"],
                "num_classes": num_classes,
                "dropout_type": config.classifier["dropout_type"],
            })
            self.classifiers.add_module(
                name, BasicClassificationHead(classifier_config))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward_gate(self, features):
        # features: (bs, layer, sent_len, hidden_size)
        B, L, S, H = features.shape
        scores = self.gate(features).squeeze(-1)  # (B, L, S)
        scores_no_noise = scores
        if self.training:  # inference only; kept for fidelity
            U = torch.rand_like(scores)
            scores += -torch.log(-torch.log(U + 1e-8) + 1e-8)
        scores /= self.gate_temperature
        if not self.training and self.fuse_ops and self.top_L >= L:
            # all captured layers are kept: skip topk+gather, the weighted
            # sum over the softmax weights is identical
            weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (B, L, S, 1)
            final_feature = (features * weights).sum(dim=1)   # (B, S, H)
            all_scores = F.softmax(scores_no_noise, dim=1)
            return final_feature, scores, None, all_scores
        topk_scores, topk_indices = torch.topk(scores, k=self.top_L, dim=1)
        expanded = topk_indices.unsqueeze(-1).expand(B, self.top_L, S, H)
        topk_features = torch.gather(features, dim=1, index=expanded)
        weights = F.softmax(topk_scores, dim=1).unsqueeze(-1)
        final_feature = (topk_features * weights).sum(dim=1)  # (B, S, H)
        all_scores = F.softmax(scores_no_noise, dim=1)
        return final_feature, topk_scores, topk_indices, all_scores

    def forward(self, features, token_type_ids=None, label_type=None,
                label_marker=None, **kwargs):
        features = self.norm(features)
        features, gate_topk_scores, gate_topk_indices, gate_all_scores = \
            self.forward_gate(features)
        features = self.input_proj(features)
        if not self.training and self.fuse_ops:
            for layer in self.adapter:
                features = fused_adapter_forward(layer, features)
            full_logits_list = classifiers_forward(self.classifiers, features)
        else:
            for layer in self.adapter:
                features = layer(features)

            full_logits_list = []
            for classifier in self.classifiers:
                full_logits_list.append(classifier(features))  # (bs, sent_len, n_cls)

        attention_mask = None
        if token_type_ids is not None:
            attention_mask = (
                torch.stack(list(token_type_ids.values()), 0).sum(0) > 0
            ).to(torch.long)

        return PGuardMultiTasksOutputForCls(
            classification_logits=None,
            full_classification_logits=full_logits_list,
            attention_mask=attention_mask,
            hidden_states=features,
            gate_topk_scores=gate_topk_scores,
            gate_topk_indices=gate_topk_indices,
            gate_all_scores=gate_all_scores,
            input_ids=kwargs.get("input_ids"),
            token_type_ids=token_type_ids,
        )

    # -- engine-facing scoring (GuardHeadBase contract) --------------------
    def score_step(self, prefill_rows, decode_rows,
                   score: bool = True) -> HeadStepResult | None:
        rows = list(prefill_rows)
        flags = [False] * len(rows)
        if score:
            # prefill rows always score; only decode rows follow the cadence
            rows += list(decode_rows)
            flags += [True] * len(decode_rows)
        if not rows:
            return None
        rows = [(rid, f[self._layer_pos]) for rid, f in rows]
        if get_config().prefill_risk_mode == "last":
            # score only each request's final token (decode rows are n=1, so
            # this is a no-op for them)
            rows = [(rid, f[:, -1:, :]) for rid, f in rows]
        if all(f.shape[1] == 1 for _, f in rows):
            # every row already has exactly one token
            padded = torch.stack([f for _, f in rows], dim=0)  # (B, C, 1, H)
        else:
            padded, _mask = pad_rows(rows)
        hidden = padded.to(dtype=self._dtype)
        with torch.no_grad():
            out = self(features=hidden, token_type_ids=None,
                       label_type=None, label_marker=None)
        probs = [torch.softmax(l.float(), dim=-1)
                 for l in out.full_classification_logits]  # (B, n, n_cls)
        ns = [f.shape[1] for _, f in rows]
        req_ids = [rid for rid, _ in rows]
        cpu = self._to_cpu_flats([[p[i, :n] for p in probs]
                                  for i, n in enumerate(ns)],
                                 req_ids, flags)
        return HeadStepResult(req_ids, cpu, ns, flags)


if __name__ == "__main__":
    # in-module self-test: python3 -m pguard_vllm.heads.lasttok
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        hidden_size=64,
        pguard_version="lasttok",
        classifier={"classification_heads": [["task_a", 3]],
                    "dropout_prob": 0.0, "dropout_type": "stable"},
        task_schema={"task_a": {"id2label": {"0": "normal", "1": "a", "2": "b"},
                                "label2id": {"normal": 0, "a": 1, "b": 2},
                                "task_id": 0}},
        selected_layers="1,2,3", label_type="single",
    )
    head = PluginGuard(cfg)
    head.eval()
    feats = torch.randn(2, 3, 5, 64)  # (bs, n_capture_layers, len, hidden)
    with torch.no_grad():
        out = head(features=feats, token_type_ids=None, label_type=None,
                   label_marker=None)
    assert len(out.full_classification_logits) == 1
    assert tuple(out.full_classification_logits[0].shape) == (2, 5, 3)
    probs = torch.softmax(out.full_classification_logits[0], dim=-1)
    assert torch.allclose(probs.sum(-1), torch.ones(2, 5), atol=1e-5)

    # engine step: mixed ragged prefill+decode rows
    head.bind(device="cpu", dtype=torch.float32, layer_pos=[0, 1, 2],
              total_cls=3, task_layout=[("task_a", 0, 3)], max_slots=4)
    pre = [("r1", torch.randn(3, 5, 64))]
    dec = [("r1", torch.randn(3, 1, 64)), ("r2", torch.randn(3, 1, 64))]
    res = head.score_step(pre, dec, score=True)
    assert res.req_ids == ["r1", "r1", "r2"]
    assert res.ns == [5, 1, 1] and res.is_decodes == [False, True, True]
    assert all(len(f) == 3 * n for f, n in zip(res.flats, res.ns))
    assert all(0.0 <= v <= 1.0 for f in res.flats for v in f)

    # cadence: decode rows dropped when score=False, prefill still scores
    res2 = head.score_step(pre, dec, score=False)
    assert res2.req_ids == ["r1"] and res2.is_decodes == [False]
    assert res2.ns == [5]

    # nothing to score -> None
    assert head.score_step([], [], score=True) is None
    assert head.score_step([], dec, score=False) is None

    # prefill_risk_mode=last: only each request's final prompt token
    import os
    os.environ["PGUARD_PREFILL_RISK_MODE"] = "last"
    try:
        reload_config()
        res3 = head.score_step(pre, [], score=True)
        assert res3.ns == [1] and res3.req_ids == ["r1"]
    finally:
        os.environ.pop("PGUARD_PREFILL_RISK_MODE", None)
        reload_config()
    print("lasttok head self-test OK (forward, engine step, cadence, "
          "prefill_risk_mode)")
