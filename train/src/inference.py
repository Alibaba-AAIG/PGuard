import os
import json
import torch
import numpy as np

from copy import deepcopy
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, AutoModelForImageTextToText

from src.model import MODEL_MAP
from src.model.modeling_pguard_ind_wrapper import PguardWrapper

from src.utils.streaming_dataset import TextStreamingDataset, VLStreamingDataset


class PostProcessor:
    def post_process_stream_full(self, inputs:dict):
        scores = inputs['scores']  # (bs, sent_len, n_head, max_dim)
        attention_mask = inputs['attention_mask']  # (bs, sent_len)
        bs = scores.shape[0]

        final_result = []

        # permute scores for convenient indexing: (bs, n_head, max_dim, sent_len)
        # the last dim is then the sequence, easy to slice with attention_mask
        scores_permuted = scores.permute(0, 2, 3, 1)

        for i in range(bs):
            # valid-token mask of this sample
            sample_mask = attention_mask[i] == 1

            # no valid tokens: emit an empty result and continue
            if not sample_mask.any():
                sent_result = {}
                for task_id, (task_name, n_classes) in enumerate(self.classification_heads):
                    sent_result[task_name] = {}
                    task_id2label = self.id2label[task_name]
                    for label_idx in range(n_classes):
                        label = task_id2label[str(label_idx)]
                        sent_result[task_name][label] = []  # empty list
                final_result.append(sent_result)
                continue

            # advanced indexing filters all valid-token scores in one shot
            # scores_permuted[i] has shape (n_head, max_dim, sent_len)
            # sample_mask has shape (sent_len)
            # result valid_scores_for_sample has shape (n_head, max_dim, num_valid_tokens)
            valid_scores_for_sample = scores_permuted[i][:, :, sample_mask]

            # one bulk CPU transfer and conversion to lists
            score_lists_for_sample = valid_scores_for_sample.cpu().tolist()

            # build the result dict efficiently
            sent_result = {}
            for task_id, (task_name, n_classes) in enumerate(self.classification_heads):
                sent_result[task_name] = {}
                task_id2label = self.id2label[task_name]
                for label_idx in range(n_classes):
                    label = task_id2label[str(label_idx)]
                    sent_result[task_name][label] = [round(x, 6) for x in score_lists_for_sample[task_id][label_idx]]  # NOTE: per-element round() is slow but keeps the serialized output compact

            final_result.append(sent_result)

        return final_result

class PluginGuardIndMultiTaskInference(PostProcessor):

    def __init__(self, feature_classifier_path, use_wrap_model:bool, max_length, feature_layers, local_rank, stream_agg,
                 llm_backend=None, tp_size=1, selected_layers=None, chunked_prefill_size=None, attention_backend=None,
                 max_pixels=None, min_pixels=None, gpu_memory_utilization=None):

        # the classifier head accepts a local directory or an HF repo id;
        # non-directory paths are resolved to a local HF snapshot directory
        if not os.path.isdir(feature_classifier_path):
            from huggingface_hub import snapshot_download
            feature_classifier_path = snapshot_download(repo_id=feature_classifier_path)

        from src.sgl.patch_sglang_all_layers import load_hf_config, ensure_dsv4_chat_template
        self.config = load_hf_config(feature_classifier_path)

        self.task_schema = self.config.task_schema
        self.id2label = {}
        for task_name in self.task_schema.keys():
            self.id2label[task_name] = self.task_schema[task_name].get('id2label')

        # VL model autodetection: Qwen3-VL / Qwen3.5-VL configs contain text_config
        self.is_vl_model = hasattr(self.config, "text_config")
        # unspecified capture settings default to the training-time values so the trained layers are selected
        if llm_backend is None:
            llm_backend = self.config.llm_backend
        if selected_layers is None:
            selected_layers = self.config.selected_layers
        if feature_layers is None:
            feature_layers = self.config.feature_layers
        self.llm_backend = llm_backend

        if self.config.pguard_version not in MODEL_MAP:
            raise NotImplementedError(self.config.pguard_version)

        self.model = MODEL_MAP[self.config.pguard_version](self.config)
        self.model.load_state_dict(torch.load(os.path.join(feature_classifier_path, "pytorch_model.bin"), map_location="cpu"))
        self.model.to(f"cuda:{local_rank}")

        if use_wrap_model:
            feature_extractor_path = self.config.feature_extractor

            if self.is_vl_model:
                # VL mode: use a processor; decode etc. go through processor.tokenizer
                self.processor = AutoProcessor.from_pretrained(feature_extractor_path, trust_remote_code=True)
                if llm_backend == "sglang" and hasattr(self.processor, "image_processor"):
                    # GLM processors ignore the max_pixels/min_pixels kwargs; overwrite
                    # image_processor.size after loading to align with the sglang side
                    # (same as train.py)
                    self.processor.image_processor.size = {
                        "longest_edge": max_pixels,
                        "shortest_edge": min_pixels,
                    }
                self.tokenizer = self.processor.tokenizer
            else:
                self.processor = None
                self.tokenizer = AutoTokenizer.from_pretrained(feature_extractor_path)
                # the official DeepSeek-V4 release ships without a chat_template;
                # inject one matching the official encoding format (same as train.py)
                ensure_dsv4_chat_template(self.tokenizer, feature_extractor_path)

            self.feature_extractor_path = feature_extractor_path

            # VL models are loaded with AutoModelForImageTextToText
            extractor_cls = AutoModelForImageTextToText if self.is_vl_model else AutoModelForCausalLM

            # choose the feature-extractor loading path by llm_backend
            if llm_backend == "hf":
                feature_extractor = extractor_cls.from_pretrained(
                    feature_extractor_path,
                    torch_dtype="auto",
                    device_map=f"cuda:{local_rank}",
                )
            elif llm_backend == "hfpp":
                feature_extractor = extractor_cls.from_pretrained(
                    feature_extractor_path,
                    torch_dtype="auto",
                    device_map="auto",
                )
            elif llm_backend == "sglang":
                from src.sgl import SGLangModel
                feature_extractor = SGLangModel(
                    model_path=feature_extractor_path,
                    max_model_len=max_length,
                    tp_size=tp_size,
                    layers=selected_layers,
                    chunked_prefill_size=chunked_prefill_size,
                    gpu_memory_utilization=gpu_memory_utilization,
                    attention_backend=attention_backend,
                    max_pixels=max_pixels,
                    min_pixels=min_pixels,
                )
            else:
                raise ValueError(f"Unknown llm_backend: {llm_backend}")

            print(f'feature_layers: {feature_layers}')

            self.model = PguardWrapper(
                feature_extractor=feature_extractor,
                feature_classifier=self.model,
                layers=feature_layers,
            )
            if llm_backend == "hf":
                self.model.device = feature_extractor.device
            else:
                self.model.device = f"cuda:{local_rank}"

            self.max_length = max_length
        else:
            raise NotImplementedError("use_wrap_model=False (embedding input) is not supported")

        # splitinf scores every token; each text content piece gets a marker
        # whose field of view covers all tokens to its left.
        self.stream_agg = stream_agg
        if stream_agg != 'splitinf':
            raise Exception(f'Unsupported stream_agg: {stream_agg}, only splitinf is supported')
        print('\n!!! streaming inference enabled !!!\n')
        self._marker_template = "<MARKER###{}>"
        self.insert_markers_in_batch_fn = self.insert_markers_in_batch
        self.dataset = self._build_marker_dataset(max_length, marker_sight=f'1,{10**10}')
        self.predict_batch = self.predict_batch_text_stream_new
        self.post_process = self.post_process_stream_full

        self.classification_heads = self.config.classifier["classification_heads"]
        self.max_dim = max([dim for _, dim in self.classification_heads])

        if self.config.label_type != 'single':
            raise NotImplementedError(f'unsupported label_type: {self.config.label_type}')
        self.activation = torch.nn.Softmax(dim=-1)

        self.model.eval()

    def _append_marker(self, text, start_id):
        """Append one marker (id = start_id + 1) to a text; returns (marked text, 1)."""
        marker_template = self._marker_template
        if not text:
            return marker_template.format(start_id + 1), 1
        return text + marker_template.format(start_id + 1), 1

    def _insert_markers_in_content(self, content, start_id=0):
        """Append one marker at the end of each text piece of a message content
        (str or multimodal list); returns (marked content, number of markers).

        A trailing non-text item (e.g. an image) gets an extra empty-text marker so
        image-only messages still carry a prediction point after the image tokens.
        """
        if isinstance(content, str):
            return self._append_marker(content, start_id)

        if isinstance(content, list):
            marker_count = 0
            marked_content = []
            for item in content:
                if item.get('type') == 'text':
                    marked_text, num_marker = self._append_marker(item.get('text', ''), start_id + marker_count)
                    marked_content.append({'type': 'text', 'text': marked_text})
                    marker_count += num_marker
                else:
                    marked_content.append(item)
            if marked_content and marked_content[-1].get('type') != 'text':
                empty_marked, num_marker = self._append_marker('', start_id + marker_count)
                marked_content.append({'type': 'text', 'text': empty_marked})
                marker_count += num_marker
            return marked_content, marker_count

        return content, 0

    def insert_markers_in_batch(self, text_list, image_list=None):
        """Build dataset items with one marker per text content piece.

        Adds an empty-text marker after a trailing non-text item. Inputs may be
        JSON strings or parsed message lists; outputs record marker roles and counts.
        """
        item_list = []
        for idx, text in enumerate(text_list):
            if isinstance(text, str):
                messages = json.loads(text)
            else:
                messages = deepcopy(text)

            # append an empty assistant turn when missing, so the response part
            # also gets a prediction point
            if not messages or messages[-1].get("role") != "assistant":
                messages.append({"role": "assistant", "content": ""})

            marked_messages = []
            marker_roles = {}
            marker_count = 0
            for msg_id, msg in enumerate(messages):
                marked_content, num_marker = self._insert_markers_in_content(
                    msg['content'], marker_count
                )
                marked_messages.append({'role': msg['role'], 'content': marked_content})
                for i in range(marker_count, marker_count+num_marker):
                    marker_roles[str(i+1)] = {'msg_id': str(msg_id), 'role': msg['role']}
                marker_count += num_marker

            label = {str(i):[] for i in range(1, marker_count+1)}

            item = {
                self.dataset.text_col: marked_messages,
                self.dataset.label_col: label,
                'marker_count': marker_count,
                'marker_roles': marker_roles,
            }
            if hasattr(self.dataset, 'image_col'):
                item[self.dataset.image_col] = image_list[idx] if image_list is not None else None
            item_list.append(item)
        return item_list

    def _build_marker_dataset(self, max_length, marker_sight):
        '''Build the text or VL marker dataset for splitinf inference.'''
        # the schema read back from config comes back in arbitrary order; re-sort by task_id
        schema = dict(sorted(self.task_schema.items(), key=lambda x: x[1]['task_id']))
        common = dict(
            max_length=max_length,
            schema_dict=schema,
            marker_sight=marker_sight,
            marker_pattern=r"<MARKER###(\d+)>",
            marker_max_id=-1,  # modified dynamically afterwards
            label_col='label',
        )
        if self.is_vl_model:
            return VLStreamingDataset(
                processor=self.processor,
                text_col='messages',
                image_col='image_url',
                **common,
            )
        return TextStreamingDataset(
            tokenizer=self.tokenizer,
            text_col='text',
            **common,
        )

    @torch.inference_mode()
    def predict_batch_text_stream_new(self, text_list, image_list=None, return_extend=False, return_gate=False):
        if isinstance(text_list, str):
            text_list = [text_list]

        # insert markers manually
        item_list = self.insert_markers_in_batch_fn(text_list, image_list)
        marker_max_id_in_batch = max(x['marker_count'] for x in item_list)

        # within one batch, every encode and the final collate must use the same marker_id_range
        self.dataset.set_marker_id_range(marker_max_id_in_batch)
        examples = [self.dataset.encoder(x) for x in item_list]
        batch = self.dataset.collate(examples)
        # indices of samples whose image failed to process (produced by the VL dataset collate)
        failed_indices = batch.get('failed_indices', [])

        device = self.model.device

        for mid in batch['batch_token_type_ids']:
            batch['batch_token_type_ids'][mid] = batch['batch_token_type_ids'][mid].to(device)

        # collate already uses the VL naming
        input_ids = batch['batch_input_ids'].to(device)
        attention_mask = batch['batch_attention_mask'].to(device)
        model_input = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=batch['batch_token_type_ids'],
            label_type=None,
            label_marker=None,
        )
        # VL-specific multimodal inputs, injected as needed
        if batch.get('batch_pixel_values') is not None:
            model_input['pixel_values'] = batch['batch_pixel_values'].to(device)
        if batch.get('batch_image_grid_thw') is not None:
            model_input['image_grid_thw'] = batch['batch_image_grid_thw'].to(device)
        # the SGLang backend produces no pixel_values; image_data must be passed directly
        # (two-level structure: [[Image]|None, ...])
        if self.llm_backend == "sglang" and batch.get('batch_images') is not None:
            model_input['image_data'] = batch['batch_images']


        outputs = self.model(**model_input)
        logits = outputs.full_classification_logits  # list of (bs, sent_len, n_classes), len=n_head

        # pad each head's n_classes up to max_dim
        scores = []
        for i in range(len(logits)):
            score = self.activation(logits[i])  # (bs, sent_len, n_classes)
            score = torch.cat([score, torch.zeros([score.shape[0], score.shape[1], self.max_dim - score.shape[-1]]).to(score.device)], dim=-1)  # (bs, sent_len, max_dim)
            scores.append(score)
        scores = torch.stack(scores, dim=2)  # (bs, sent_len, n_head, max_dim)

        # The model may filter and re-pad masks, so input and output sent_len can
        # differ. Scores and the output mask must nevertheless stay aligned.
        assert scores.shape[1] == outputs.attention_mask.shape[1]

        res = {
            'scores': scores,
            'attention_mask': outputs.attention_mask,  # (bs, sent_len); summing over markers makes it independent of marker order
            'token_type_ids': outputs.token_type_ids,  # (marker, bs, sent_len); dim 0 follows batch_token_type_ids.keys()
            'failed_indices': failed_indices,
        }

        if return_extend:
            bs = scores.shape[0]
            extend_fields = [{} for _ in range(bs)]

            for i in range(bs):
                extend_fields[i].update(item_list[i])
                extend_fields[i]['marker_max_id_in_batch'] = marker_max_id_in_batch

            if 'input_ids' in outputs:
                for i in range(bs):
                    extend_fields[i]['marker_input_ids'] = outputs.input_ids[i][outputs.attention_mask[i] == 1].cpu().tolist()
                    extend_fields[i]['marker_input_tokens'] = [self.tokenizer.decode(x) for x in extend_fields[i]['marker_input_ids']]

            if 'token_type_ids' in outputs:
                marker_role_mask = []
                for a, b, c in zip(outputs.token_type_ids.transpose(1,0), outputs.attention_mask, item_list):   # token_type_ids(marker, bs, sent_len) is stacked in batch_token_type_ids.keys order
                    t = {}
                    used_mask = None  # masks out already-used positions so role masks do not overlap, for compatibility with stream_agg=='splitinf' where slice_len=inf and marker_sight='1,inf'
                    for idx,mid in enumerate(c['marker_roles']):
                        t[mid] = a[idx, b==1]
                        if used_mask is None:
                            used_mask = torch.zeros_like(t[mid])
                        t[mid] = t[mid] * (used_mask == 0)
                        used_mask += (t[mid] > 0)
                        t[mid] = t[mid].cpu().tolist()

                    marker_role_mask.append(t)

                for i in range(bs):
                    extend_fields[i]['marker_role_mask'] = marker_role_mask[i]

            if return_gate and 'gate_all_scores' in outputs:
                gate_all_scores = [a[:, b==1] for a, b in zip(outputs.gate_all_scores, outputs.attention_mask)]  # scores: (bs, n_layer, sent_len), attn: (bs, sent_len) -> list of (n_layer, sent_len)
                gate_all_scores = [a.transpose(1,0) for a in gate_all_scores]  # list of (layer, sent_len) -> list of (sent_len, layer)
                gate_all_scores = [a.cpu().tolist() for a in gate_all_scores]
                gate_all_scores = [[[round(c,6) for c in b] for b in a] for a in gate_all_scores]

                for i in range(bs):
                    extend_fields[i]['gate_all_scores'] = gate_all_scores[i]

            res['extend_fields'] = extend_fields

        return res


    def parse_result_to_different_roles(self, result, marker_roles, marker_role_mask):
        '''Group per-token scores by {msg_id}_{role}.

        Each group uses the union of its marker masks. Marker ids increase from
        left to right before this function is called.
        '''
        id_role2marker_ids = {}
        id_role2marker_mask = {}

        for mid, v in marker_roles.items():
            id_role = v['msg_id'] + '_' + v['role']

            if id_role not in id_role2marker_ids:
                id_role2marker_ids[id_role] = []
            id_role2marker_ids[id_role].append(mid)

            if id_role not in id_role2marker_mask:
                id_role2marker_mask[id_role] = []
            id_role2marker_mask[id_role].append(marker_role_mask[mid])

        result_sep = {}

        for id_role, marker_mask in id_role2marker_mask.items():
            marker_mask = np.array(marker_mask).sum(0) > 0
            res = deepcopy(result)
            for head, label_score in res.items():
                for label, scores in label_score.items():
                    res[head][label] = np.array(scores)[marker_mask].tolist()
            result_sep[id_role] = res

        return result_sep

    def predict(self, text_or_embedding, image_list=None, return_extend=False, return_gate=False):
        # Bound methods must be compared with == because each attribute access
        # creates a new bound object.
        if self.predict_batch == self.predict_batch_text_stream_new:
            predict_result = self.predict_batch(text_or_embedding, image_list=image_list, return_extend=return_extend, return_gate=return_gate)  # dict
        else:
            predict_result = self.predict_batch(text_or_embedding, return_extend=return_extend, return_gate=return_gate)  # dict
        post_result = self.post_process(predict_result)  # list

        # indices of samples whose image failed to process (skipped when writing results)
        failed_indices = predict_result.get('failed_indices', [])

        if 'extend_fields' in predict_result:
            assert len(post_result) == len(predict_result['extend_fields'])

            final_res = []
            for a,b in zip(post_result, predict_result['extend_fields']):
                res = {'result': a, 'extend_fields': b}

                # also split results so each [msgid]_[role] gets its own algo_result
                if b.get('marker_roles') is not None and b.get('marker_role_mask') is not None:
                    res['result_sep'] = self.parse_result_to_different_roles(result=a, marker_roles=b['marker_roles'], marker_role_mask=b['marker_role_mask'])

                final_res.append(res)

            return final_res, failed_indices
        else:
            return post_result, failed_indices
