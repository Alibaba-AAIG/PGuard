import itertools
import json
import re
from typing import Optional, Union

import torch
from PIL import Image, ImageFile
from torch.utils.data import IterableDataset

# Critical: let PIL tolerate truncated images, otherwise an OSError can be raised
# inside the DataLoader feeder thread (pickle.dumps -> __getstate__ -> tobytes() ->
# load()), aborting the whole DataLoader outside any user try/except.
ImageFile.LOAD_TRUNCATED_IMAGES = True

from src.utils.template_utils import create_marker_mask, create_marker_mask_vl, get_generation_prompt_ids_vl


class BaseStreamingDataset(IterableDataset):
    '''Optional sharded JSONL iteration plus marker encoding and collation.'''

    def __init__(self,
            max_length: int,
            schema_dict: dict,
            marker_pattern=r"<MARKER###(\d+)>",
            marker_max_id=1,  # marker id range is [1, max_id]
            marker_sight='2,512,1,10',  # alternating value,length pairs; all values >0 contribute to loss
            overlap_mode='none',  # how overlapping marker masks are resolved: none / front / back

            table_path=None,  # local jsonl path; one JSON object per line
            slice_id=0,
            slice_count=1,
            text_col='text',
            label_col='label',
        ):
        super().__init__()

        self.max_length = max_length
        self.marker_sight = marker_sight
        self.marker_pattern = marker_pattern
        self.overlap_mode = overlap_mode
        self.set_marker_id_range(marker_max_id)
        self.text_col = text_col
        self.label_col = label_col

        self.schema_dict = schema_dict

        self.task_num_classes = {}
        for idx, (task_name, task_detail) in enumerate(self.schema_dict.items()):
            self.task_num_classes[task_detail['task_id']] = len(task_detail['label2id'])
            assert idx == task_detail['task_id'], 'this must hold, otherwise multitask_labels breaks'

        # max number of classes over all tasks
        self.task_max_classes = max(self.task_num_classes.values())

        if table_path:
            # local jsonl metadata: total line count and column names (keys of the first row)
            self.table_path = table_path
            with open(table_path, 'r', encoding='utf-8') as f:
                self.total_row_count = sum(1 for _ in f)
                f.seek(0)
                first_line = f.readline()
            self.input_columns = list(json.loads(first_line).keys()) if self.total_row_count > 0 else []
            self.start_pos, self.end_pos = self._get_slice_range(self.total_row_count, slice_id, slice_count)
            self.row_count = self.end_pos - self.start_pos

            assert self.text_col in self.input_columns
            assert self.label_col in self.input_columns

            print("jsonl total_row_count:{}, slice:[{},{}), row_count:{}, columns:{}".format(
                self.total_row_count, self.start_pos, self.end_pos, self.row_count, self.input_columns))

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        print("worker_id:{}, num_workers:{}".format(worker_id, num_workers))

        # further split this rank's slice line range across workers
        start_idx, end_idx = self._get_slice_range(self.row_count, worker_id, num_workers, baseline=self.start_pos)
        print(f"reading lines {start_idx} to {end_idx} from {self.table_path}")

        with open(self.table_path, 'r', encoding='utf-8') as f:
            for line in itertools.islice(f, start_idx, end_idx):
                data = json.loads(line)
                encoded = self.encoder(data)
                assert encoded is not None, "encoder must not return None: a None return makes batch counts inconsistent across ranks and causes DDP NCCL collective timeouts"
                yield encoded

    def _get_slice_range(self, row_count, worker_id, num_workers, baseline=0):
        # div-mod split, each slice data count max diff 1
        size = int(row_count / num_workers)
        split_point = row_count % num_workers
        if worker_id < split_point:
            start = worker_id * (size + 1) + baseline
            end = start + (size + 1)
        else:
            start = split_point * (size + 1) + (worker_id - split_point) * size + baseline
            end = start + size
        return start, end

    def __len__(self):
        return self.row_count

    def set_marker_id_range(self, marker_max_id):
        self.marker_id_range = range(1, max(marker_max_id+1, 2))

    def multitask_labels(self, dt_labels, data_type, schema_dict, n_classes):
        multitask_labels_list = []
        label_mask = []

        dt_list = list(schema_dict.keys())
        for dt_type in dt_list:
            labels = torch.zeros(n_classes, dtype=torch.float)  # all-zero when this task is unsupervised
            if dt_type == data_type:
                labels[dt_labels] = 1
                label_mask.append(1.0)
            else:
                label_mask.append(0.0)
            multitask_labels_list.append(labels.tolist())
        return multitask_labels_list, label_mask

    def encode_labels(self, raw_label):
        """Parse task###label strings into per-task label vectors and head masks."""
        all_multitask_labels_list = []
        all_label_mask = []
        head_label_dic = {}

        if isinstance(raw_label, str):
            tiled_labels = json.loads(raw_label)
        else:
            tiled_labels = raw_label

        for head_label in tiled_labels:
            head, label = head_label.split('###')
            if head not in head_label_dic:
                head_label_dic[head] = []
            if label.lower() in ('null', 'none', 'nan', 'empty', ' '):
                raise Exception(f'null label is not allowed for single-label training: {tiled_labels}')
            head_label_dic[head].append(label)

        for task_name, labels in head_label_dic.items():

            if len(labels) != 1:
                raise Exception(f'head {task_name} is not single-label: {tiled_labels}')

            labels = [self.schema_dict[task_name]['label2id'][x] for x in labels]

            multitask_labels_list, label_mask = self.multitask_labels(labels, task_name, self.schema_dict, self.task_max_classes)

            # multitask_labels_list: (len(schema_dict), task_max_classes), single-label vectors
            # label_mask: (len(schema_dict),)

            all_multitask_labels_list.append(multitask_labels_list)
            all_label_mask.append(label_mask)

        # all_multitask_labels_list: (len(head_label_dic), len(schema_dict), task_max_classes)
        # all_label_mask: list, (len(head_label_dic), len(schema_dict))

        if head_label_dic:
            all_multitask_labels_list = torch.tensor(all_multitask_labels_list, dtype=torch.bool).sum(0).long().tolist()
            all_label_mask = torch.tensor(all_label_mask, dtype=torch.bool).sum(0).long().tolist()
        else:
            all_multitask_labels_list = [[0]*self.task_max_classes for _ in range(len(self.schema_dict))]
            all_label_mask = [0] * len(self.schema_dict)

        # all_multitask_labels_list: (len(schema_dict), task_max_classes)
        # all_label_mask: (len(schema_dict),)

        return tiled_labels, head_label_dic, all_multitask_labels_list, all_label_mask

    def collate_once(self, examples):
        batch_raw_labels, batch_task_label, batch_task_mask = [], [], []

        task_num = len(examples[0]['multitask_labels_list'])
        batch_multi_labels = [list() for i in range(task_num)]

        for item in examples: # batch_size
            batch_raw_labels.append(item['raw_label'])
            batch_task_label.append(item['task_label'])
            batch_task_mask.append(item['task_mask'])

            for idx, multitask_label in enumerate(item['multitask_labels_list']):
                batch_multi_labels[idx].append(multitask_label)

        batch_task_mask = torch.tensor(batch_task_mask).float()

        for idx, item in enumerate(batch_multi_labels):
            batch_multi_labels[idx] = torch.tensor((batch_multi_labels[idx])).float()

            n_classes = self.task_num_classes[idx]
            batch_multi_labels[idx] = batch_multi_labels[idx][:, :n_classes]  # (bs, max_classes) -> (bs, num_classes)

        # Single-label vectors -> class indices; all-zero vectors use index 0 and are masked out.
        batch_single_labels = [x.argmax(1) for x in batch_multi_labels]

        return {
            'batch_raw_labels': batch_raw_labels,
            'batch_task_label': batch_task_label,
            'batch_task_mask': batch_task_mask,
            'batch_single_labels': batch_single_labels,
        }


class TextStreamingDataset(BaseStreamingDataset):
    '''Text dataset: applies the tokenizer chat template and builds per-marker
    token masks via create_marker_mask.'''

    def __init__(self,
            tokenizer,
            max_length: int,
            schema_dict: dict,
            marker_pattern=r"<MARKER###(\d+)>",
            marker_max_id=1,
            marker_sight='2,512,1,10',
            overlap_mode='none',
            table_path=None,
            slice_id=0,
            slice_count=1,
            text_col='text',
            label_col='label',
        ):
        super().__init__(
            max_length=max_length,
            schema_dict=schema_dict,
            marker_pattern=marker_pattern,
            marker_max_id=marker_max_id,
            marker_sight=marker_sight,
            overlap_mode=overlap_mode,
            table_path=table_path,
            slice_id=slice_id,
            slice_count=slice_count,
            text_col=text_col,
            label_col=label_col,
        )
        self.tokenizer = tokenizer

    def encoder(self, item):
        try:
            if isinstance(item[self.text_col], str):
                messages = json.loads(item[self.text_col])
            else:
                messages = item[self.text_col]

            if isinstance(item[self.label_col], str):
                labels = {int(k): v for k, v in json.loads(item[self.label_col]).items()}
            else:
                labels = {int(k): v for k, v in item[self.label_col].items()}

            label_marker = {}
            for id in self.marker_id_range:
                tiled_labels, head_label_dic, all_multitask_labels_list, all_label_mask = self.encode_labels(labels.get(id, []))
                label_marker[id] = {
                    'raw_label': tiled_labels,  # flattened labels
                    'task_label': list(head_label_dic.keys()),  # heads involved
                    'task_mask': all_label_mask,  # head mask
                    'multitask_labels_list': all_multitask_labels_list,  # labels concatenated over all heads
                }

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as e:
            print(f'data encoding failed, emitting placeholder data (excluded from loss): {e}, raw data: {item}')
            # emit a minimal valid placeholder so that per-rank sample counts stay consistent
            label_marker = {}
            for id in self.marker_id_range:
                tiled_labels, head_label_dic, all_multitask_labels_list, all_label_mask = self.encode_labels([])
                label_marker[id] = {
                    'raw_label': tiled_labels,
                    'task_label': list(head_label_dic.keys()),
                    'task_mask': all_label_mask,  # encode_labels([]) returns all-zero, excluded from loss
                    'multitask_labels_list': all_multitask_labels_list,
                }
            placeholder_messages = [{'role': 'user', 'content': 'placeholder'}]
            try:
                text = self.tokenizer.apply_chat_template(placeholder_messages, tokenize=False, add_generation_prompt=False)
            except Exception:
                text = "placeholder"
            return {
                'text': text,
                'label_marker': label_marker,
            }

        return {
            'text': text,
            'label_marker': label_marker,
        }

    def collate(self, examples: list[dict]):
        batch_text = [item['text'] for item in examples]

        model_inputs = create_marker_mask(
            texts=batch_text,
            tokenizer=self.tokenizer,
            tokenizer_config=dict(padding=True, truncation=True, max_length=self.max_length, return_tensors="pt", padding_side='left'),
            n_left=self.marker_sight,
            marker_pattern=self.marker_pattern,
            overlap_mode=self.overlap_mode,
        )
        batch_input_ids = model_inputs['input_ids']
        batch_attention_mask = model_inputs['attention_mask']
        batch_token_type_ids = {}
        for id in self.marker_id_range:
            batch_token_type_ids[id] = model_inputs['marker_masks'].pop(id, torch.zeros_like(batch_input_ids, dtype=torch.long))

        if model_inputs['marker_masks']:
            raise Exception(f"marker ids outside the preset id range {self.marker_id_range}: {list(model_inputs['marker_masks'].keys())}")

        batch_label_marker = {}
        for id in self.marker_id_range:
            _examples: list[dict] = [item['label_marker'][id] for item in examples]
            batch_label_marker[id] = self.collate_once(_examples)

        assert len(batch_token_type_ids) == len(batch_label_marker), 'labels must correspond one-to-one'
        for a, b in zip(batch_token_type_ids.keys(), batch_label_marker.keys()):
            assert a == b, 'labels must correspond one-to-one'

        return {
            'batch_text': batch_text,
            'batch_input_ids': batch_input_ids,
            'batch_attention_mask': batch_attention_mask,
            'batch_token_type_ids': batch_token_type_ids,
            'batch_label_marker': batch_label_marker,
        }


class VLStreamingDataset(BaseStreamingDataset):
    '''VL dataset: multimodal content lists, image loading (local path or URL)
    and per-marker token masks via create_marker_mask_vl, plus the
    assistant_start query/response boundary for heads that need it.'''

    def __init__(self,
            processor,
            max_length: int,
            schema_dict: dict,
            marker_pattern=r"<MARKER###(\d+)>",
            marker_max_id=1,
            marker_sight='2,512,1,10',
            overlap_mode='none',
            table_path=None,
            slice_id=0,
            slice_count=1,
            text_col='text',
            label_col='label',
            image_col='image',  # image column; can be a local path or a URL
            max_pixels=448*448,
            min_pixels=224*224,
        ):
        super().__init__(
            max_length=max_length,
            schema_dict=schema_dict,
            marker_pattern=marker_pattern,
            marker_max_id=marker_max_id,
            marker_sight=marker_sight,
            overlap_mode=overlap_mode,
            table_path=table_path,
            slice_id=slice_id,
            slice_count=slice_count,
            text_col=text_col,
            label_col=label_col,
        )
        self.processor = processor
        self.image_col = image_col
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

        # Used to locate the last chat-template generation prompt in collate.
        self.generation_prompt_ids = get_generation_prompt_ids_vl(processor)

    def to_rgb(self, pil_image: Image.Image) -> Image.Image:
        if pil_image.mode == "RGBA":
            white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
            white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
            return white_background
        else:
            return pil_image.convert("RGB")

    def load_image(self, image_path: Union[str, None]) -> Optional[Image.Image]:
        """Load an image from a local path or URL; returns None on failure."""
        if image_path is None or image_path == '':
            return None

        try:
            if image_path.startswith('http://') or image_path.startswith('https://'):
                # load from URL
                import requests
                from io import BytesIO
                response = requests.get(image_path, timeout=30)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            else:
                # load from local path
                image = Image.open(image_path)

            # convert to RGB (handles RGBA, P and other modes)
            if image.mode != 'RGB':
                image = self.to_rgb(image)
            return image
        except Exception as e:
            print(f"failed to load image: {image_path}, error: {e}")
            return None

    def encoder(self, item):
        try:
            data_invalid = False  # invalid samples get their task_mask zeroed

            if isinstance(item[self.text_col], str):
                messages = json.loads(item[self.text_col])
            else:
                messages = item[self.text_col]

            # ensure every message's content is a list
            for x in messages:
                if isinstance(x['content'], str):
                    x['content'] = [{'type': 'text', 'text': x['content']}]

            if isinstance(item[self.label_col], str):
                labels = {int(k): v for k, v in json.loads(item[self.label_col]).items()}
            else:
                labels = {int(k): v for k, v in item[self.label_col].items()}

            # image_col may be None (text-only training / inference) or the field may be
            # empty; both fall into the text-only branch
            if self.image_col and item.get(self.image_col):
                if isinstance(item[self.image_col], str):
                    image_path = item[self.image_col]
                    # load the image up front for validation; failure marks the sample invalid
                    image = self.load_image(image_path)
                    if image is None:
                        print(f"image failed to load, marking sample as invalid: {image_path}")
                        data_invalid = True
                        # drop image-type contents from messages so the processor does not
                        # expect image data; supports both HF (type='image') and
                        # OpenAI (type='image_url') formats
                        for x in messages:
                            x['content'] = [c for c in x['content'] if c.get('type') not in ('image', 'image_url')]
                    else:
                        # make sure the first user message has a type:image entry
                        # the image content only serves as a placeholder for apply_chat_template;
                        # the actual image is loaded via image_col
                        # NOTE: check for an existing image-type content instead of len==1
                        # (upstream messages[0]['content'] may already be [{'type':'image',...}],
                        # length 1 but with the placeholder present)
                        # supports both HF (type='image') and OpenAI (type='image_url') formats
                        has_image = any(c.get('type') in ('image', 'image_url') for c in messages[0]['content'])
                        if not has_image:
                            messages[0]['content'].insert(0, {'type': 'image', 'image': 'xxx'})
                else:
                    raise ValueError("multiple images per sample are not supported")
            else:
                image = None

            # clean vision special tokens out of the text and mark the sample invalid
            # (never return None, to keep per-rank sample counts consistent)
            for x in messages:
                for c in x['content']:
                    if isinstance(c, dict) and c.get('type') == 'text' and isinstance(c.get('text'), str):
                        if re.search(r'<\|(image_pad|vision_start|vision_end|video_pad)\|>', c['text']):
                            print(f"text contains vision special tokens, cleaning and marking sample invalid: {c['text'][:100]}")
                            c['text'] = re.sub(r'<\|(image_pad|vision_start|vision_end|video_pad)\|>', '', c['text'])
                            data_invalid = True

            label_marker = {}
            for id in self.marker_id_range:
                tiled_labels, head_label_dic, all_multitask_labels_list, all_label_mask = self.encode_labels(labels.get(id, []))
                # for invalid samples (image load failure / vision special tokens / other errors),
                # zero the task_mask so they do not contribute to the loss backward
                if data_invalid:
                    all_label_mask = [0] * len(all_label_mask)
                label_marker[id] = {
                    'raw_label': tiled_labels,  # flattened labels
                    'task_label': list(head_label_dic.keys()),  # heads involved
                    'task_mask': all_label_mask,  # head mask
                    'multitask_labels_list': all_multitask_labels_list,  # labels concatenated over all heads
                }

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

        except Exception as e:
            print(f'data encoding failed, emitting placeholder data (excluded from loss): {e}, raw data: {item}')
            # emit a minimal valid placeholder so that per-rank sample counts stay consistent
            label_marker = {}
            for id in self.marker_id_range:
                tiled_labels, head_label_dic, all_multitask_labels_list, all_label_mask = self.encode_labels([])
                label_marker[id] = {
                    'raw_label': tiled_labels,
                    'task_label': list(head_label_dic.keys()),
                    'task_mask': all_label_mask,  # encode_labels([]) returns all-zero, excluded from loss
                    'multitask_labels_list': all_multitask_labels_list,
                }
            placeholder_messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'placeholder'}]}]
            try:
                text = self.processor.apply_chat_template(placeholder_messages, tokenize=False, add_generation_prompt=False)
            except Exception:
                text = "placeholder"
            return {
                'messages': text,
                'label_marker': label_marker,
                'images': None,
            }

        return {
            'messages': text,
            'label_marker': label_marker,
            'images': image,  # None or the image
        }

    def collate(self, examples: list[dict]):
        """Collate rendered messages and one PIL image or None per sample."""
        batch_messages = []
        batch_images = []

        for item in examples:
            batch_messages.append(item['messages'])
            batch_images.append(item['images'])

        model_inputs = create_marker_mask_vl(
            messages_list=batch_messages,
            batch_images=batch_images,
            processor=self.processor,
            processor_config=dict(
                padding=True,
                truncation=True,
                max_length=self.max_length,
                padding_side='left',
                max_pixels=self.max_pixels,
                min_pixels=self.min_pixels,
            ),
            n_left=self.marker_sight,
            marker_pattern=self.marker_pattern,
            overlap_mode=self.overlap_mode,
            generation_prompt_ids=self.generation_prompt_ids,
        )

        batch_input_ids = model_inputs['input_ids']
        batch_attention_mask = model_inputs['attention_mask']
        batch_pixel_values = model_inputs.get('pixel_values', None)
        batch_image_grid_thw = model_inputs.get('image_grid_thw', None)
        batch_effective_images = model_inputs['effective_images']
        # Last generation-prompt start in padded coordinates, (bs,)
        batch_assistant_start = model_inputs['assistant_start']

        batch_token_type_ids = {}
        for id in self.marker_id_range:
            batch_token_type_ids[id] = model_inputs['marker_masks'].pop(id, torch.zeros_like(batch_input_ids, dtype=torch.long))

        if model_inputs['marker_masks']:
            raise Exception(f"marker ids outside the preset id range {self.marker_id_range}: {list(model_inputs['marker_masks'].keys())}")

        batch_label_marker = {}
        for id in self.marker_id_range:
            _examples: list[dict] = [item['label_marker'][id] for item in examples]
            batch_label_marker[id] = self.collate_once(_examples)

        # zero the task_mask of samples whose image failed to load (equivalent to
        # IGNORE_INDEX: no loss is computed for them)
        failed_indices = model_inputs.get('failed_indices', [])
        if failed_indices:
            print(f"{len(failed_indices)} samples in this batch failed image processing; their task_mask is zeroed to skip loss computation: {failed_indices}")
            for id in self.marker_id_range:
                batch_label_marker[id]['batch_task_mask'][failed_indices, :] = 0

        assert len(batch_token_type_ids) == len(batch_label_marker), 'labels must correspond one-to-one'
        for a, b in zip(batch_token_type_ids.keys(), batch_label_marker.keys()):
            assert a == b, 'labels must correspond one-to-one'

        # image_data for SGLang uses a two-level structure: [[Image], None, [Image], ...]
        # so sglang io_struct._normalize_image_data can correctly distinguish the
        # modalities of None samples and avoid tagging text-only samples as "image"
        # in mixed batches.
        batch_images_sglang = [
            [img] if img is not None else None for img in batch_effective_images
        ]

        return {
            'batch_text': batch_messages,
            'batch_input_ids': batch_input_ids,
            'batch_attention_mask': batch_attention_mask,
            'batch_pixel_values': batch_pixel_values,
            'batch_image_grid_thw': batch_image_grid_thw,
            'batch_images': batch_images_sglang,
            'batch_token_type_ids': batch_token_type_ids,
            'batch_assistant_start': batch_assistant_start,
            'batch_label_marker': batch_label_marker,
            'failed_indices': failed_indices,
        }
