import torch
import torch.nn as nn


def process_feature(layers, input_ids, attention_mask, token_type_ids, hidden_states: list[torch.Tensor], **kwargs) -> dict:
    """Reduce captured backbone hidden states into the head input.

    ``layers`` is ``"all"`` or a list of indices into the captured
    representations (accepts a comma-separated str or list[int]); with
    ``num_hidden_layers`` transformer layers there are ``num_hidden_layers + 1``
    captured representations (0 = embedding output, 1..N-1 = residual stream
    after each layer, N = final norm output), and ``"all"`` selects 1..N-1,
    skipping the embedding and the final norm. Only the selected
    representations are kept, in the given order. The full sequence is always
    passed through: the marker masks (token_type_ids) supervise output
    positions / the loss and never filter tokens, and the real
    ``attention_mask`` is forwarded so stateful heads (e.g. EMA) can recurse
    over all real tokens.
    """
    num_hidden_layers = len(hidden_states) - 1
    if isinstance(layers, str):
        if layers.strip().lower() == 'all':
            layers = list(range(1, num_hidden_layers))
        else:
            try:
                layers = [int(x) for x in layers.split(',') if x.strip() != '']
            except ValueError:
                raise ValueError(f'layers must be "all" or a comma-separated int list, got: {layers!r}')
    else:
        layers = list(layers)
    if not layers:
        raise ValueError('layers must resolve to at least one layer index')
    bad = [x for x in layers if not 0 <= x <= num_hidden_layers]
    if bad:
        raise ValueError(f'layer indices out of range [0, {num_hidden_layers}]: {bad}')

    hidden_states = torch.stack(hidden_states, 1)  # (bs, layer, sent_len, hidden_dim)
    hidden_states = hidden_states[:, layers, :, :]  # (bs, n_layer, sent_len, hidden_dim)

    return {
        'features': hidden_states,
        'token_type_ids': token_type_ids,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'label_type': kwargs.get('label_type'),
        'label_marker': kwargs.get('label_marker'),
    }


class PguardWrapper(nn.Module):
    """Wraps a frozen feature extractor (HF model or SGLangModel) with a
    trainable PlugGuard classification head; extracts hidden states, selects
    the configured layers, and forwards the result to the head."""

    def __init__(self, feature_extractor, feature_classifier, layers, extractor_eval=True):
        """``layers`` is the feature-layer spec resolved by process_feature
        ("all" or indices into the captured representations)."""
        super().__init__()

        self.feature_extractor = feature_extractor
        self.feature_classifier = feature_classifier
        self.layers = layers

        self.extractor_eval = extractor_eval

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        image_data=None,
        **kwargs,
    ):
        if isinstance(self.feature_extractor, nn.Module):
            if self.extractor_eval:
                self.feature_extractor.eval()
            else:
                self.feature_extractor.train()

        # extra multimodal inputs for VL models, injected as needed
        extractor_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
        }
        if pixel_values is not None:
            extractor_inputs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            extractor_inputs["image_grid_thw"] = image_grid_thw
        if pixel_values_videos is not None:
            extractor_inputs["pixel_values_videos"] = pixel_values_videos
        if video_grid_thw is not None:
            extractor_inputs["video_grid_thw"] = video_grid_thw
        if image_data is not None:
            extractor_inputs["image_data"] = image_data

        with torch.no_grad():
            outputs = self.feature_extractor(**extractor_inputs)

            # under PP (device_map="auto"), hidden_states of different layers may
            # live on different GPUs; move them all to the classifier's device to
            # avoid cross-device errors in the later torch.stack
            # for non-PP modes .to(same_device) is a no-op with no extra cost
            classifier_device = next(self.feature_classifier.parameters()).device
            classifier_dtype = next(self.feature_classifier.parameters()).dtype
            hidden_states = tuple(hs.to(classifier_device) for hs in outputs.hidden_states)

            try:
                feature_dict: dict = process_feature(
                    layers=self.layers,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    hidden_states=hidden_states,
                    **kwargs,
                )
            except Exception as e:
                raise Exception(f'process_feature failed: {e}')

            feature_dict['features'] = feature_dict['features'].to(classifier_dtype)

        # assistant_start is the GRU query/response boundary: the start of the
        # last chat-template generation prompt, in padded input coordinates.
        if kwargs.get('assistant_start') is not None:
            feature_dict['assistant_start'] = kwargs['assistant_start']

        extra_args = {
            'output_attentions': kwargs.get('output_attentions', False),
        }
        result = self.feature_classifier(**feature_dict, **extra_args)

        if kwargs.get('return_feature_dict'):
            return result, feature_dict
        else:
            return result
