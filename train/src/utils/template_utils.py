import re
from typing import Dict, List, Union

import torch
from transformers import PreTrainedTokenizer


def create_marker_mask(
    texts: List[str],
    tokenizer: PreTrainedTokenizer,
    tokenizer_config: dict,
    n_left: int|dict,
    marker_pattern: str = r"<MARKER###(\d+)>",
    overlap_mode: str = "none",
) -> Dict[str, Union[torch.Tensor, Dict[int, torch.Tensor]]]:
    """Tokenize texts and build one left-context mask per marker id.

    n_left is an integer, a {value: length} dict, or alternating value,length
    pairs such as '2,512,1,10'; later regions overwrite earlier ones.
    marker_pattern must capture an integer id. overlap_mode resolves overlaps
    with smaller-id-wins (front), larger-id-wins (back), or no resolution (none).

    Returns tokenizer outputs plus marker_masks keyed by integer id. Each mask
    has input_ids shape: unmarked positions are 0 and marked positions contain
    the configured positive value.
    """
    if overlap_mode not in ("none", "front", "back"):
        raise ValueError(f"overlap_mode must be 'none', 'front', or 'back', got '{overlap_mode}'")
    if isinstance(n_left, int):
        n_left = {1:n_left}
    elif isinstance(n_left, str):
        n_left = {int(k):int(v) for k,v in zip(n_left.split(',')[::2], n_left.split(',')[1::2])}
    if min(n_left.keys()) <= 0 or min(n_left.values()) <= 0:
        raise ValueError("n_left must be a positive integer.")

    all_markers_info = []
    cleaned_texts = []

    for text in texts:
        markers = []
        for match in re.finditer(marker_pattern, text):
            markers.append({
                'id': int(match.group(1)),
                'char_pos': match.start(),
                'text_len': len(match.group(0))
            })

        if len(markers) > len(set([x['id'] for x in markers])):
            raise Exception(f'text contains duplicated marker ids: {text}')

        markers.sort(key=lambda m: m['char_pos'])

        cumulative_removed_len = 0
        for marker in markers:
            marker['cleaned_pos'] = marker['char_pos'] - cumulative_removed_len
            cumulative_removed_len += marker['text_len']

        all_markers_info.append(markers)
        cleaned_texts.append(re.sub(marker_pattern, "", text))

    tokenizer_config['return_tensors'] = 'pt'
    tokenizer_config['return_offsets_mapping'] = True
    tokenizer_output = tokenizer(cleaned_texts, **tokenizer_config)

    input_ids = tokenizer_output["input_ids"]
    attention_mask = tokenizer_output["attention_mask"]
    offset_mapping = tokenizer_output["offset_mapping"]

    marker_masks = {}

    for i in range(len(texts)):
        sample_markers = all_markers_info[i]
        sample_offsets = offset_mapping[i]

        # map every char covered by a token (not just offset[0]): chars on both
        # sides of a marker can merge into one token, and a start-offset-only map
        # yields an all-zero mask in that case
        char_to_token_map = {}
        for token_idx, offset in enumerate(sample_offsets):
            start_char = offset[0].item()
            end_char = offset[1].item()
            if start_char < end_char:
                # map every character index covered by this token to token_idx
                for char_idx in range(start_char, end_char):
                    char_to_token_map[char_idx] = token_idx

        for marker in sample_markers:
            marker_id = marker['id']
            cleaned_pos = marker['cleaned_pos']

            # first time this marker_id is seen: create an all-zero mask tensor for it
            if marker_id not in marker_masks:
                marker_masks[marker_id] = torch.zeros_like(input_ids, dtype=torch.long)

            target_token_idx = char_to_token_map.get(cleaned_pos)

            # edge case: the marker sits at the very end of the cleaned text (no
            # characters after it). Typical scenario: chat templates like GLM's that
            # append the marker directly at the end of the assistant segment without
            # a trailing <|im_end|>-style token. char_to_token_map then has no entry
            # for cleaned_pos; fall back to "one past the last valid token" so that
            # [target-n_left, target) still covers the trailing n_left real tokens,
            # matching the normal-case semantics.
            if target_token_idx is None and char_to_token_map and cleaned_pos >= len(cleaned_texts[i]):
                target_token_idx = max(char_to_token_map.values()) + 1

            if target_token_idx is not None:
                for m, n in n_left.items():
                    mask_start_idx = max(0, target_token_idx - n)
                    mask_end_idx = target_token_idx

                    # set the region on this marker_id's mask
                    marker_masks[marker_id][i, mask_start_idx:mask_end_idx] = m

    if overlap_mode != "none" and len(marker_masks) > 1:
        # front: smaller ids take priority; back: larger ids take priority
        sorted_ids = sorted(marker_masks.keys(), reverse=(overlap_mode == "back"))
        # occupied: positions already taken by a higher-priority marker
        occupied = torch.zeros_like(input_ids, dtype=torch.bool)
        for mid in sorted_ids:
            # zero out positions of the current marker's mask that are already occupied
            overlap = occupied & (marker_masks[mid] > 0)
            if overlap.any():
                marker_masks[mid][overlap] = 0
            # add the current marker's non-zero positions to occupied
            occupied = occupied | (marker_masks[mid] > 0)

    for marker_id in marker_masks:
        marker_masks[marker_id] = marker_masks[marker_id] * attention_mask.long()

    del tokenizer_output["offset_mapping"]
    tokenizer_output["marker_masks"] = marker_masks

    return tokenizer_output


from PIL import Image


def get_generation_prompt_ids_vl(processor) -> List[int]:
    '''Return chat-template generation prompt ids.

    For Qwen this is typically [151644, 77091, 198] (<|im_start|>assistant\\n).
    '''
    def _to_id_list(tokens):
        # normalize BatchEncoding / dict / Tensor / list[list] return shapes
        if hasattr(tokens, 'input_ids') or isinstance(tokens, dict):
            tokens = tokens['input_ids']
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        if tokens and isinstance(tokens[0], (list, tuple)):
            tokens = list(tokens[0])
        return list(tokens)

    # NOTE: VL processors (e.g. Qwen3-VL) require content in the structured list
    # format; a plain string raises
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'test'}]}]
    tokens1 = _to_id_list(processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
    tokens2 = _to_id_list(processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
    assert all([a == b for a, b in zip(tokens1, tokens2)])
    return tokens2[len(tokens1):]


def find_subtensor_last_indices(xs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    For each row of the 2D tensor xs, find the start index of the LAST occurrence
    of the 1D tensor y.

    Args:
        xs (torch.Tensor): the 2D tensor to search in, (N, M).
        y (torch.Tensor): the 1D tensor to search for, length L.

    Returns:
        torch.Tensor: shape (N,); element i is the start index of the last
            occurrence of y in row i of xs, or -1 if not found.
    """
    if xs.dim() != 2 or y.dim() != 1:
        raise ValueError("input tensors xs must be 2D and y must be 1D")

    num_rows, row_len = xs.shape
    len_y = y.numel()
    if len_y == 0 or len_y > row_len:
        return torch.full((num_rows,), -1, dtype=torch.long, device=xs.device)

    y = y.to(dtype=xs.dtype, device=xs.device)
    # xs: (N, M) -> windows: (N, M-L+1, L)
    windows = xs.unfold(dimension=1, size=len_y, step=1)
    matches = torch.all(windows == y, dim=2)              # (N, M-L+1)
    has_match = torch.any(matches, dim=1)                 # (N,)
    num_windows = matches.shape[1]
    # fliplr + argmax trick to find the last True
    last = (num_windows - 1) - torch.argmax(torch.fliplr(matches).int(), dim=1)
    return torch.where(has_match, last, torch.full_like(last, -1))


def create_marker_mask_vl(
    messages_list: List[List[dict]],
    batch_images,
    processor,
    processor_config: dict,
    n_left: int|dict,
    marker_pattern: str = r"<MARKER###(\d+)>",
    overlap_mode: str = "none",
    generation_prompt_ids: List[int] = None,
) -> Dict[str, Union[torch.Tensor, Dict[int, torch.Tensor]]]:
    """Process rendered chat-template strings and build per-marker VL masks.

    batch_images contains one PIL image or None per sample. n_left, marker_pattern,
    and overlap_mode have the same semantics as create_marker_mask. Optional
    generation_prompt_ids locate the last generation prompt; they are computed
    from the processor when omitted.

    Returns input_ids, attention_mask, marker_masks, failed_indices,
    effective_images, assistant_start, and available VL extras. assistant_start
    is a (batch_size,) tensor in padded input coordinates; unmatched samples get
    0, selecting a fully sequential zero initial state in the GRU head.
    """
    if overlap_mode not in ("none", "front", "back"):
        raise ValueError(f"overlap_mode must be 'none', 'front', or 'back', got '{overlap_mode}'")
    if isinstance(n_left, int):
        n_left = {1: n_left}
    elif isinstance(n_left, str):
        n_left = {int(k): int(v) for k, v in zip(n_left.split(',')[::2], n_left.split(',')[1::2])}
    if min(n_left.keys()) <= 0 or min(n_left.values()) <= 0:
        raise ValueError("n_left must be a positive integer.")

    tokenizer = processor.tokenizer
    padding_side = processor_config.get('padding_side', 'left')
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    batch_results = []
    failed_indices = []
    effective_images = []
    for i, (text, image) in enumerate(zip(messages_list, batch_images)):
        # extract marker info
        markers = []
        for match in re.finditer(marker_pattern, text):
            markers.append({
                'id': int(match.group(1)),
                'char_pos': match.start(),
                'text_len': len(match.group(0))
            })

        if len(markers) > len(set([x['id'] for x in markers])):
            raise Exception(f'text contains duplicated marker ids: {text}')

        markers.sort(key=lambda m: m['char_pos'])

        # build the clean text
        cleaned_text = re.sub(marker_pattern, "", text)

        # recompute marker positions within cleaned_text
        # so cleaned_pos shares the coordinate system with offset_mapping
        cleaned_markers = []
        for marker in markers:
            # find the position in cleaned_text (the text preceding the marker)
            original_prefix = text[:marker['char_pos']]
            cleaned_prefix = re.sub(marker_pattern, "", original_prefix)
            # the marker's position in cleaned_text is len(cleaned_prefix)
            marker['cleaned_pos'] = len(cleaned_prefix)
            cleaned_markers.append(marker)

        # process this sample
        if image is not None:
            # image-text sample
            sample_inputs = {
                'text': [cleaned_text],
                'images': [image],
                'return_tensors': 'pt',
            }
            sample_inputs.update(processor_config)
            try:
                sample_output = processor(**sample_inputs)
                effective_images.append(image)
            except Exception as e:
                print(f"processor failed on an image-text sample, retrying with a mock image: {e}")
                mock_image = Image.new("RGB", (224, 224), (255, 255, 255))
                sample_inputs['images'] = [mock_image]
                try:
                    sample_output = processor(**sample_inputs)
                    effective_images.append(mock_image)
                except Exception as e2:
                    # after truncation the text may contain no image tokens, so even
                    # the mock image fails; fall back to text-only mode
                    print(f"mock image failed too, falling back to text-only mode: {e2}")
                    sample_inputs.pop('images', None)
                    # NOTE: the image placeholder in text must be removed in sync
                    # Qwen: <|vision_start|><|image_pad|><|vision_end|>
                    # GLM:   <|begin_of_image|>...<|end_of_image|>
                    # otherwise processor._check_special_mm_tokens sees image tokens
                    # in text but not in the truncated input_ids and raises a Mismatch
                    sample_inputs['text'] = [
                        re.sub(r'<\|vision_start\|>.*?<\|vision_end\|>|<\|begin_of_image\|>.*?<\|end_of_image\|>', '', t, flags=re.DOTALL)
                        for t in sample_inputs['text']
                    ]
                    sample_output = processor(**sample_inputs)
                    effective_images.append(None)
                failed_indices.append(i)
            input_ids_sample = sample_output['input_ids'][0]
            attention_mask_sample = sample_output['attention_mask'][0]

            # keep VL-specific outputs
            vl_extras = {}
            for key in ['pixel_values', 'image_grid_thw', 'pixel_values_videos', 'video_grid_thw']:
                if key in sample_output:
                    vl_extras[key] = sample_output[key]
        else:
            # text-only sample
            sample_inputs = {
                'text': [cleaned_text],
                'return_tensors': 'pt',
            }
            sample_inputs.update(processor_config)
            sample_output = processor(**sample_inputs)
            input_ids_sample = sample_output['input_ids'][0]
            attention_mask_sample = sample_output['attention_mask'][0]
            vl_extras = {}
            effective_images.append(None)

        # get offset_mapping via the tokenizer (needed for both image-text and text-only)
        tokenizer_config = {
            'padding': False,
            'truncation': processor_config.get('truncation', True),
            'max_length': processor_config.get('max_length', 8192),
            'return_offsets_mapping': True,
            'return_tensors': 'pt',
        }
        tokenizer_output = tokenizer([cleaned_text], **tokenizer_config)
        offset_mapping_sample = tokenizer_output['offset_mapping'][0]
        text_input_ids_sample = tokenizer_output['input_ids'][0]

        # build the char -> token mapping (in cleaned_text coordinates)
        char_to_text_token = {}
        for token_idx, offset in enumerate(offset_mapping_sample):
            start_char = offset[0].item()
            end_char = offset[1].item()
            if start_char < end_char:
                for char_idx in range(start_char, end_char):
                    char_to_text_token[char_idx] = token_idx

        batch_results.append({
            'input_ids': input_ids_sample,
            'attention_mask': attention_mask_sample,
            'text_input_ids': text_input_ids_sample,
            'markers': markers,
            'char_to_text_token': char_to_text_token,
            'has_image': image is not None,
            'vl_extras': vl_extras,
        })

    # find the max sequence length
    max_len = max(r['input_ids'].shape[0] for r in batch_results)

    # create batch tensors
    batch_size = len(batch_results)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    # collect VL-specific outputs
    vl_extras_batch = {}

    # fill the batch
    for i, result in enumerate(batch_results):
        seq_len = result['input_ids'].shape[0]
        if padding_side == 'left':
            start_pos = max_len - seq_len
            input_ids[i, start_pos:] = result['input_ids']
            attention_mask[i, start_pos:] = result['attention_mask']
        else:
            input_ids[i, :seq_len] = result['input_ids']
            attention_mask[i, :seq_len] = result['attention_mask']

        # collect VL-specific outputs
        for key, value in result['vl_extras'].items():
            if key not in vl_extras_batch:
                vl_extras_batch[key] = []
            vl_extras_batch[key].append(value)

    # merge VL-specific outputs (if any)
    for key in vl_extras_batch:
        if key == 'pixel_values':
            vl_extras_batch[key] = torch.cat(vl_extras_batch[key], dim=0)
        elif key == 'image_grid_thw':
            vl_extras_batch[key] = torch.cat(vl_extras_batch[key], dim=0)
        # other fields handled as needed

    # Start of the last generation prompt in padded input coordinates, not the
    # first assistant content token. Unmatched samples clamp to 0 (no query).
    if generation_prompt_ids is None:
        generation_prompt_ids = get_generation_prompt_ids_vl(processor)
    assistant_start = find_subtensor_last_indices(
        input_ids, torch.tensor(generation_prompt_ids, dtype=torch.long)
    ).clamp(min=0)

    # get the image token id
    image_token_id = None
    if hasattr(processor, 'image_token_id'):
        image_token_id = processor.image_token_id
    elif hasattr(tokenizer, 'image_token_id'):
        image_token_id = tokenizer.image_token_id
    else:
        for token_str in ['<image>', '<|image_pad|>', '<|image|>', '<|vision_start|>', '<|vision_end|>']:
            if token_str in tokenizer.get_vocab():
                image_token_id = tokenizer.convert_tokens_to_ids(token_str)
                break

    # init marker_masks
    marker_masks = {}

    # generate the marker mask for each sample
    for i, result in enumerate(batch_results):
        markers = result['markers']
        char_to_text_token = result['char_to_text_token']
        text_input_ids_sample = result['text_input_ids']
        input_ids_sample = result['input_ids']
        has_image = result['has_image']

        # find the actual content boundaries (excluding padding)
        non_pad_mask = result['attention_mask'].bool()
        if non_pad_mask.any():
            full_seq_start = non_pad_mask.nonzero(as_tuple=True)[0][0].item()
            full_seq_end = non_pad_mask.nonzero(as_tuple=True)[0][-1].item() + 1
        else:
            full_seq_start = 0
            full_seq_end = input_ids_sample.shape[0]

        # find image token positions
        image_positions = set()
        if image_token_id is not None:
            image_positions = set((input_ids_sample == image_token_id).nonzero(as_tuple=True)[0].tolist())

        # build the text_token_idx -> full_token_idx mapping
        text_to_full_mapping = {}
        if has_image:
            # with an image: the mapping must skip image tokens
            text_seq = text_input_ids_sample.tolist()
            full_seq = input_ids_sample.tolist()

            text_idx = 0
            for full_idx in range(full_seq_start, full_seq_end):
                if full_seq[full_idx] == image_token_id:
                    continue
                if text_idx < len(text_seq):
                    text_to_full_mapping[text_idx] = full_idx
                    text_idx += 1
        else:
            # text-only: 1:1 mapping
            for text_idx in range(len(text_input_ids_sample)):
                text_to_full_mapping[text_idx] = full_seq_start + text_idx

        for marker in markers:
            marker_id = marker['id']
            cleaned_pos = marker['cleaned_pos']

            # find the text_token_idx of the marker
            if has_image:
                lookup_pos = cleaned_pos - 1 if cleaned_pos > 0 else 0
            else:
                lookup_pos = cleaned_pos
            text_token_idx = char_to_text_token.get(lookup_pos)

            if text_token_idx is None:
                for pos in range(lookup_pos, -1, -1):
                    if pos in char_to_text_token:
                        text_token_idx = char_to_text_token[pos]
                        break
            if text_token_idx is None:
                for pos in range(cleaned_pos, max(char_to_text_token.keys()) + 1):
                    if pos in char_to_text_token:
                        text_token_idx = char_to_text_token[pos]
                        break
            if text_token_idx is None:
                continue

            # map to full_token_idx
            full_token_idx = text_to_full_mapping.get(text_token_idx)
            if full_token_idx is None:
                continue

            # init the mask
            if marker_id not in marker_masks:
                marker_masks[marker_id] = torch.zeros((batch_size, max_len), dtype=torch.long)

            # compute the mask range (positions within the batch)
            for m, n in n_left.items():
                if padding_side == 'left':
                    batch_full_token_idx = max_len - (full_seq_end - full_seq_start) + (full_token_idx - full_seq_start)
                else:
                    batch_full_token_idx = full_token_idx

                base_start_idx = max(0, batch_full_token_idx - n)
                base_end_idx = batch_full_token_idx

                # check for intersecting image tokens
                batch_image_positions = set()
                if padding_side == 'left':
                    offset = max_len - (full_seq_end - full_seq_start)
                    for pos in image_positions:
                        batch_image_positions.add(offset + (pos - full_seq_start))
                else:
                    batch_image_positions = image_positions

                intersecting = [pos for pos in batch_image_positions
                               if base_start_idx <= pos < base_end_idx
                               or pos == base_start_idx - 1
                               or pos == base_end_idx]

                if intersecting:
                    min_intersect = min(intersecting)
                    max_intersect = max(intersecting)

                    mask_start_idx = min_intersect
                    while mask_start_idx > 0 and mask_start_idx - 1 in batch_image_positions:
                        mask_start_idx -= 1

                    mask_end_idx = max_intersect + 1
                    while mask_end_idx < max_len and mask_end_idx in batch_image_positions:
                        mask_end_idx += 1

                    mask_start_idx = min(mask_start_idx, base_start_idx)
                    mask_end_idx = max(mask_end_idx, base_end_idx)
                    mask_start_idx = max(0, mask_start_idx)
                    mask_end_idx = min(max_len, mask_end_idx)
                else:
                    mask_start_idx = base_start_idx
                    mask_end_idx = base_end_idx

                marker_masks[marker_id][i, mask_start_idx:mask_end_idx] = m

    # resolve mask overlap between markers
    if overlap_mode != "none" and len(marker_masks) > 1:
        # front: smaller ids take priority; back: larger ids take priority
        sorted_ids = sorted(marker_masks.keys(), reverse=(overlap_mode == "back"))
        # occupied: positions already taken by a higher-priority marker
        occupied = torch.zeros_like(input_ids, dtype=torch.bool)
        for mid in sorted_ids:
            # zero out positions of the current marker's mask that are already occupied
            overlap = occupied & (marker_masks[mid] > 0)
            if overlap.any():
                marker_masks[mid][overlap] = 0
            # add the current marker's non-zero positions to occupied
            occupied = occupied | (marker_masks[mid] > 0)

    # apply attention_mask to marker_masks
    for marker_id in marker_masks:
        marker_masks[marker_id] = marker_masks[marker_id] * attention_mask

    # build the return value
    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'marker_masks': marker_masks,
        'assistant_start': assistant_start,
        'failed_indices': failed_indices,
        'effective_images': effective_images,
    }
    result.update(vl_extras_batch)

    return result
