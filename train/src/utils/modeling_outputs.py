from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers.modeling_outputs import ModelOutput


@dataclass
class XguardBaseModelOutputWithAdapter(ModelOutput):
    """Backbone and adapter outputs with optional pooling and transformer intermediates."""

    last_hidden_state: torch.FloatTensor = None
    adapter_hidden_state: torch.FloatTensor = None

    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class PGuardMultiTasksOutputForCls(ModelOutput):
    """Multi-task token-classification outputs.

    full_classification_logits contains one (batch, sequence, num_classes)
    tensor per task. Other fields carry marker masks, gate/fusion diagnostics,
    losses, and optional intermediate states.
    """

    classification_logits: torch.FloatTensor = None
    full_classification_logits: torch.FloatTensor = None
    attention_mask: torch.LongTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    backbone_last_hidden_state: torch.FloatTensor = None
    gate_topk_scores: torch.FloatTensor = None
    gate_topk_indices: torch.LongTensor = None
    gate_all_scores: torch.FloatTensor = None
    gate_all_indices: torch.LongTensor = None
    loss: torch.Tensor = None
    loss_dict: dict = None
    input_ids: torch.Tensor = None
    token_type_ids: torch.Tensor = None
    fusion_attentions: torch.FloatTensor = None
