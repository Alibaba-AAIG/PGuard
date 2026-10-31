import torch
import torch.nn as nn
from transformers.activations import ACT2FN

# The normal (non-risk) label name; its id must be 0 in every task schema.
NORMAL_LABEL = "normal"


class _DictConfig:
    """Attribute-access view over a flat dict; supports `in` for key checks."""

    def __init__(self, d):
        self.__dict__.update(d)

    def __contains__(self, key):
        return key in self.__dict__


def config_from_dict(d):
    """Build an attribute-access config object from a flat dict."""
    return _DictConfig(d)


class BasicClassificationHead(nn.Module):
    """Token-level classification head."""

    def __init__(self, config):
        super().__init__()

        if "dropout_type" not in config:
            dropout_type = "stable"
        else:
            dropout_type = config.dropout_type

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        if dropout_type == 'stable':
            self.dropout = StableDropout(config.dropout_prob)
        else:
            self.dropout = nn.Dropout(config.dropout_prob)

        self.out_proj = nn.Linear(config.hidden_size, config.num_classes)

        self.act_fn = ACT2FN['gelu']

    def forward(self, feature, **kwargs):
        # feature: (..., hidden_size)
        feature = self.dense(feature)
        feature = self.act_fn(feature)
        feature = self.dropout(feature)
        logits = self.out_proj(feature)

        return logits


class DropoutContext(object):
    def __init__(self):
        self.dropout = 0
        self.mask = None
        self.scale = 1
        self.reuse_mask = True


def get_mask(input, local_context):
    if not isinstance(local_context, DropoutContext):
        dropout = local_context
        mask = None
    else:
        dropout = local_context.dropout
        dropout *= local_context.scale
        mask = local_context.mask if local_context.reuse_mask else None

    if dropout > 0 and mask is None:
        mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).bool()

    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask

    return mask, dropout


class XDropout(torch.autograd.Function):
    """Optimized dropout function to save computation and memory by using mask operation instead of multiplication."""

    @staticmethod
    def forward(ctx, input, local_ctx):
        mask, dropout = get_mask(input, local_ctx)
        ctx.scale = 1.0 / (1 - dropout)
        if dropout > 0:
            ctx.save_for_backward(mask)
            return input.masked_fill(mask, 0) * ctx.scale
        else:
            return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scale > 1:
            (mask,) = ctx.saved_tensors
            return grad_output.masked_fill(mask, 0) * ctx.scale, None
        else:
            return grad_output, None


class StableDropout(nn.Module):
    """Dropout with optional reusable masks."""

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.count = 0
        self.context_stack = None

    def forward(self, x):
        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())
        return x

    def clear_context(self):
        self.count = 0
        self.context_stack = None

    def init_context(self, reuse_mask=True, scale=1):
        if self.context_stack is None:
            self.context_stack = []
        self.count = 0
        for c in self.context_stack:
            c.reuse_mask = reuse_mask
            c.scale = scale

    def get_context(self):
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())
            ctx = self.context_stack[self.count]
            ctx.dropout = self.drop_prob
            self.count += 1
            return ctx
        else:
            return self.drop_prob


def count_parameters(model: "torch.nn.Module"):
    """Return trainable and total parameter counts."""
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by itemsize
        if param.__class__.__name__ == "Params4bit":
            if hasattr(param, "quant_storage") and hasattr(param.quant_storage, "itemsize"):
                num_bytes = param.quant_storage.itemsize
            elif hasattr(param, "element_size"):  # for older pytorch version
                num_bytes = param.element_size()
            else:
                num_bytes = 1

            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def show_trainable_parameters(model):
    trainable_params, all_param = count_parameters(model)
    param_stats = "trainable params: {:,} || all params: {:,} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param
    )
    print(param_stats)
