import os
import ast
import json
import torch
import logging
import math
from collections import OrderedDict
import torch.nn as nn
import torch.utils.data.distributed
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForImageTextToText
from torch import cuda
from torch.nn.parallel import DistributedDataParallel

from transformers import HfArgumentParser
from transformers import get_linear_schedule_with_warmup
from src.utils.argument import TrainArguments


from src.model import MODEL_MAP
from src.model.modeling_pguard_ind_wrapper import PguardWrapper
from src.utils.model_utils import NORMAL_LABEL

from src.utils.streaming_dataset import TextStreamingDataset, VLStreamingDataset
from src.utils.model_utils import show_trainable_parameters
from src.utils.ddp_utils import init_ddp, all_gather, dist_barrier, is_dist_avail_and_initialized

from tqdm import tqdm

logging.basicConfig(format='%(asctime)s -[PlugGuard]-[%(name)s]-[%(levelname)s] - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def _iterable_batch_count(total_rows, num_workers, batch_size):
    """Count batches using the dataset's div-mod worker sharding.

    Each worker yields its own tail batch, so estimating from total rows can
    truncate training early (10 rows / 8 workers / bs=2 yields 8 batches, not 5).
    """
    num_workers = max(num_workers, 1)
    size = total_rows // num_workers
    split_point = total_rows % num_workers
    batch_count = 0
    for worker_id in range(num_workers):
        rows = size + 1 if worker_id < split_point else size
        if rows > 0:
            batch_count += math.ceil(rows / batch_size)
    return batch_count


# Training procedure for independent PlugGuard head
def dist_train_plugin_guard_ind(args):

    if args.version not in MODEL_MAP:
        raise ValueError(f'Unsupported --version: {args.version}, supported values: {list(MODEL_MAP.keys())}')
    if args.label_type != 'single':
        raise ValueError(f'Unsupported --label_type: {args.label_type}, only single is supported')

    if args.dataset_dir is not None:
        dataset_info_path = os.path.join(args.dataset_dir, "dataset_info.json")
        with open(dataset_info_path, "r", encoding="utf-8") as file:
            dataset_info = json.load(file)
    else:
        try:
            dataset_info = json.loads(args.dataset_info.strip('"\''))
        except:
            dataset_info = ast.literal_eval(args.dataset_info.strip('"\''))

    if not args.use_wrap_model:
        raise Exception("[PlugGuard] --use_wrap_model must be set to True.")

    if RANK == 0:
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        else:
            if not args.overwrite:
                raise Exception("[PlugGuard] The directory already exists. Be careful not to overwrite any files.")

    schema_dict = OrderedDict(dataset_info)
    training_tasks = list(schema_dict.keys())

    task_schema = {}
    classification_heads = []
    for idx, task in enumerate(training_tasks):
        label2id = dataset_info[task]["label2id"]
        id2label = {v: k for k, v in label2id.items()}

        _task_schema = {
            "label2id": label2id,
            "id2label": id2label,
            "task_id": idx
        }
        task_schema[task] = _task_schema

        schema_dict[task]["id2label"] = id2label
        schema_dict[task]["task_id"] = idx

        classification_heads.append((task, len(label2id)))

        for k, v in dataset_info[task]["label2id"].items():
            if k == NORMAL_LABEL and v != 0:
                raise Exception(f'the id of the normal label ({NORMAL_LABEL}) must be 0')

    classifier_config = {
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "dropout_prob": args.dropout_prob,
        "dropout_type": args.dropout_type,
        "classification_heads": classification_heads
    }


    from src.sgl.patch_sglang_all_layers import load_hf_config, ensure_dsv4_chat_template
    config = load_hf_config(args.model_name_or_path)

    # detect text vs VL model
    is_vl_model = hasattr(config, "text_config")
    if is_vl_model:
        config.hidden_size = config.text_config.hidden_size
        config.num_hidden_layers = config.text_config.num_hidden_layers
        model_class = AutoModelForImageTextToText
        tokenizer = AutoProcessor.from_pretrained(args.model_name_or_path)
        if args.llm_backend == "sglang" and hasattr(tokenizer, "image_processor"):
            # GLM processors silently ignore the max_pixels/min_pixels kwargs; the
            # expanded image token count is decided by image_processor.size, so we
            # overwrite that attribute after loading to align with the sglang side
            # (reading the model-dir file directly is avoided due to download timing).
            # Qwen behaves the same as the existing kwargs path.
            tokenizer.image_processor.size = {
                "longest_edge": args.max_pixels,
                "shortest_edge": args.min_pixels,
            }
        logger.info("[VL] Detected VL model, using AutoProcessor + AutoModelForImageTextToText")
    else:
        model_class = AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side='left')
        # the official DeepSeek-V4 release ships without a chat_template; inject one
        # matching the official encoding format
        ensure_dsv4_chat_template(tokenizer, args.model_name_or_path)
        logger.info("[Text] Detected text model, using AutoTokenizer + AutoModelForCausalLM")

    config.classifier = classifier_config
    config.task_schema = task_schema
    config.label_type = args.label_type
    config.pguard_version = args.version
    config.job_name = args.job_name
    config.job_args = str(args)
    config.feature_extractor = args.model_name_or_path
    config.feature_layers = args.feature_layers
    config.selected_layers = args.selected_layers
    config.llm_backend = args.llm_backend

    dtype = torch.float32
    device = f'cuda:{LOCAL_RANK}' if cuda.is_available() else 'cpu'

    pguard_model = MODEL_MAP[args.version](config)
    pguard_model = pguard_model.to(dtype)  # .to(device) happens later

    if args.llm_backend == "sglang":
        # Each DP rank owns one TP engine and a DDP classifier. Each process must
        # see exactly tp_size GPUs; the classifier uses cuda:LOCAL_RANK.
        tp_size = args.tp_size
        device_count = torch.cuda.device_count()
        assert tp_size == device_count, (
            f"[SGLang] tp_size({tp_size}) != device_count({device_count}); "
            f"make sure each process sees exactly tp_size GPUs (e.g. via CUDA_VISIBLE_DEVICES)"
        )

        # Import and create SGLangModel as feature_extractor
        from src.sgl import SGLangModel

        feature_extractor = SGLangModel(
            model_path=args.model_name_or_path,
            max_model_len=args.max_length,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tp_size=tp_size,
            layers=args.selected_layers,
            chunked_prefill_size=args.chunked_prefill_size,
            attention_backend=args.attention_backend,
            max_pixels=args.max_pixels,
            min_pixels=args.min_pixels,
            engine_extra_args=({"max_prefill_tokens": args.max_prefill_tokens} if args.max_prefill_tokens else None),
        )

    elif args.llm_backend == "hfpp":
        # Pipeline parallel mode: device_map="auto" spreads model layers over the visible GPUs
        feature_extractor = model_class.from_pretrained(
            args.model_name_or_path,
            torch_dtype="auto",
            device_map="auto",
        )
        # freeze backbone encoder
        for name, parameters in feature_extractor.named_parameters():
            parameters.requires_grad = False

    elif args.llm_backend == "hf":
        feature_extractor = model_class.from_pretrained(
            args.model_name_or_path,
            torch_dtype="auto",
            low_cpu_mem_usage=True,  # reduce peak CPU memory while loading
        )

        # freeze backbone encoder
        for name, parameters in feature_extractor.named_parameters():
            parameters.requires_grad = False


    ### pguard_wrapper init ###
    pguard_model = PguardWrapper(
        feature_extractor=feature_extractor,
        feature_classifier=pguard_model,
        layers=args.feature_layers,
    )

    logger.info("Built PlugGuard wrapper")

    show_trainable_parameters(pguard_model)

    ### ddp setup ###
    dist_barrier()

    if args.llm_backend == "sglang":
        # SGLang mode: the SGLang engine manages its own TP GPUs; the classifier goes
        # to the GPU indexed by LOCAL_RANK (LOCAL_RANK = RANK % device_count, matching
        # the NCCL cudaDev assignment; see ddp_utils.py)
        pguard_model.feature_classifier = pguard_model.feature_classifier.to(device)
        pguard_model.feature_classifier = DistributedDataParallel(
            pguard_model.feature_classifier,
            device_ids=[LOCAL_RANK],
            find_unused_parameters=True,
        )
        pguard_model_without_ddp = pguard_model.feature_classifier.module
    elif args.llm_backend == "hfpp":
        # PP mode: feature_extractor is spread over GPUs via device_map; the classifier
        # goes to the LOCAL_RANK GPU on its own
        pguard_model.feature_classifier = pguard_model.feature_classifier.to(device)
        pguard_model.feature_classifier = DistributedDataParallel(
            pguard_model.feature_classifier,
            device_ids=[LOCAL_RANK],
            find_unused_parameters=True,
        )
        pguard_model_without_ddp = pguard_model.feature_classifier.module
    elif args.llm_backend == "hf":
        pguard_model.to(device)
        # DDP construction broadcasts parameters, so all ranks must arrive here together
        dist_barrier()
        pguard_model = DistributedDataParallel(pguard_model, device_ids=[LOCAL_RANK], find_unused_parameters=True)
        pguard_model_without_ddp = pguard_model.module.feature_classifier

    if RANK == 0:
        config.save_pretrained(args.output_path)

    ### training dataset setup ###
    common_kwargs = dict(
        max_length=args.max_length,
        schema_dict=schema_dict,
        marker_pattern=args.marker_pattern,
        marker_max_id=args.marker_max_id,
        marker_sight=args.marker_sight,
        overlap_mode=args.overlap_mode,
        table_path=args.input_table,
        slice_id=RANK,
        slice_count=WORLD_SIZE,
        text_col=args.text_col,
        label_col=args.label_col,
    )
    if is_vl_model:
        # VL training: VLStreamingDataset takes a processor; image_col is optional
        # (omitting it means text-only training, which still works on a VL model)
        DatasetCls = VLStreamingDataset
        common_kwargs.update(
            processor=tokenizer,  # for VL models the tokenizer is actually an AutoProcessor
            image_col=args.image_col,  # may be None, meaning text-only training
            max_pixels=args.max_pixels,
            min_pixels=args.min_pixels,
        )
    else:
        DatasetCls = TextStreamingDataset
        common_kwargs.update(
            tokenizer=tokenizer,
        )
    train_data = DatasetCls(**common_kwargs)
    torch.distributed.barrier()

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=False,
        sampler=None,
        num_workers=args.input_table_worker,
        collate_fn=train_data.collate,
    )

    ### optimizer and scheduler setup ###
    # true batch count: deterministically computed with the dataset's div-mod worker
    # sharding formula
    data_size = _iterable_batch_count(
        len(train_data), num_workers=args.input_table_worker, batch_size=args.batch_size)

    # check data_size consistency across ranks: uneven step counts desynchronize training
    all_data_sizes = all_gather(data_size)
    ds_max, ds_min = max(all_data_sizes), min(all_data_sizes)
    # with a 1-row difference between ranks, the true batch count may still differ by
    # more than 1 due to per-worker tail batches (a legitimate distribution); align to the minimum
    if ds_max - ds_min > 1:
        logger.warning(f'[RANK={RANK}] data_size of this rank={data_size}, data_sizes of all ranks={all_data_sizes}, aligning all ranks to {ds_min}')
    data_size = ds_min

    num_training_steps = math.ceil(data_size * args.epochs / args.accumulation_steps)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in pguard_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,  # weight decay for parameters other than bias / LayerNorm
        },
        {
            "params": [p for n, p in pguard_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,   # no weight decay for bias / LayerNorm parameters
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=int(num_training_steps * 0.1),
    )

    total_data = sum(all_gather(len(train_data)))  # sum of per-rank row counts
    logger.info(f'slice data size: {len(train_data)}, total data size: {total_data}, step per epoch: {data_size}, total train steps: {num_training_steps}')

    total_loss = 0.0
    n_steps_all = 0
    last_grad_norm = 0.0
    error_count = 0

    dist_barrier()

    ### training loop ###
    for epoch in range(args.epochs):
        total_loss = 0
        n_steps = 0
        pguard_model.train()
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Training epoch={epoch}', mininterval=10)):

            # align to the minimum data_size across ranks: a rank with more data would
            # otherwise run extra steps and desynchronize training
            if batch_idx >= data_size:
                print(f'[RANK={RANK}] this rank reached data_size={data_size}, skipping remaining batches')
                break

            pguard_model.train()

            for mid, item in batch['batch_label_marker'].items():
                item['batch_task_mask'] = item['batch_task_mask'].to(device)
                if item.get('batch_single_labels'):
                    for idx, _ in enumerate(item['batch_single_labels']):
                        item['batch_single_labels'][idx] = item['batch_single_labels'][idx].to(device)

            for mid in batch['batch_token_type_ids']:
                batch['batch_token_type_ids'][mid] = batch['batch_token_type_ids'][mid].to(device)

            model_input = dict(
                input_ids=batch['batch_input_ids'].to(device),
                attention_mask=batch['batch_attention_mask'].to(device),
                token_type_ids=batch['batch_token_type_ids'],
                label_type=args.label_type,
                label_marker=batch['batch_label_marker'],
            )

            # multimodal inputs specific to VL models, injected as needed
            # SGLang needs raw PIL images (image_data); HF needs pixel_values/image_grid_thw
            if args.llm_backend == "sglang":
                if batch.get('batch_images') is not None:
                    model_input['image_data'] = batch['batch_images']
            else:
                if batch.get('batch_pixel_values') is not None:
                    model_input['pixel_values'] = batch['batch_pixel_values'].to(device)
                if batch.get('batch_image_grid_thw') is not None:
                    model_input['image_grid_thw'] = batch['batch_image_grid_thw'].to(device)

            # The extractor forward precedes the DDP classifier. Heads must not add
            # registered buffers, SyncBatchNorm, or collectives: a rank-specific
            # extractor failure would otherwise desynchronize NCCL collectives.
            local_skip = 0
            try:
                outputs = pguard_model(**model_input)
            except Exception as e:
                local_skip = 1
                if "hidden_states length mismatch with input" in str(e):
                    print(f"[RANK={RANK}] WARNING: SGLang hidden_states length mismatch, skipping this batch "
                          f"(epoch={epoch}, batch_idx={batch_idx}): {e}", flush=True)
                else:
                    print(f'[RANK={RANK}] pguard forward failed at batch_idx={batch_idx}: {e}', flush=True)
                    error_count += 1
                    if error_count > 10:
                        raise

            # cross-rank sync: backward is a synchronization point, so a failure on any
            # rank must abort the batch on all ranks
            skip_tensor = torch.tensor([local_skip], device=device, dtype=torch.int32)
            dist.all_reduce(skip_tensor, op=dist.ReduceOp.MAX)
            if skip_tensor.item() > 0:
                print(f"[RANK={RANK}] WARNING: model forward failed on {'this' if local_skip else 'another'} rank, all ranks skip batch_idx={batch_idx}", flush=True)
                continue

            error_count = 0

            final_loss = outputs['loss'] / args.accumulation_steps

            final_loss.backward()

            if (batch_idx + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == data_size:
                grad_norm = nn.utils.clip_grad_norm_(
                    parameters=pguard_model.parameters(),
                    max_norm=1.0,
                )
                last_grad_norm = grad_norm.item()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += final_loss.item()
            n_steps += 1
            n_steps_all += 1

            # logging
            if n_steps_all % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]

                metric_logs = {"epoch": epoch+batch_idx/data_size, 'n_steps':n_steps_all, "loss": outputs['loss'].item(), "smooth-loss": total_loss/n_steps, "grad_norm": last_grad_norm, "lr": current_lr}

                if 'loss_dict' in outputs:
                    loss_dict = {k:v.item() for k,v in outputs['loss_dict'].items() if k not in metric_logs}
                    metric_logs.update(loss_dict)

                if outputs.get('gate_all_scores') is not None:
                    gate_scores = outputs['gate_all_scores']  # (bs, layer, sent_len)
                    avg_gates = gate_scores.mean(dim=[0, 2])  # (layer)
                    for l_idx, score in enumerate(avg_gates):
                        metric_logs[f"gate/layer_{str(l_idx).zfill(2)}_mean"] = score.item()

                print(','.join([" {}={}".format(k, round(v,4) if isinstance(v, (int,float)) else v) for k,v in metric_logs.items()]))

                if RANK == 0:  # only rank 0 writes the log
                    with open(os.path.join(args.output_path, 'metric.log'), 'a') as f:
                        f.write(json.dumps(metric_logs, ensure_ascii=False) + '\n')

        print(f'epoch {epoch} completed !')
        if RANK == 0:
            print(f'saving epoch {epoch} model...')
            torch.save(pguard_model_without_ddp.state_dict(), os.path.join(args.output_path, '{0}.pth.{1}'.format(args.model_name, epoch)))
            torch.save(pguard_model_without_ddp.state_dict(), os.path.join(args.output_path, 'pytorch_model.bin'))
        dist_barrier()

    print('training completed !')
    if RANK == 0:
        print('saving final model...')
        torch.save(pguard_model_without_ddp.state_dict(), os.path.join(args.output_path, 'pytorch_model.bin'))
    dist_barrier()

    # Cleanup SGLang engine if used
    if args.llm_backend == "sglang":
        logger.info("[SGLang] Shutting down SGLang engine...")
        # pguard_model is PguardWrapper, its feature_extractor is SGLangModel
        pguard_model.feature_extractor.shutdown()
        logger.info("[SGLang] Engine shutdown complete.")

    # tear down the NCCL process group to terminate the HeartbeatMonitor thread
    if WORLD_SIZE > 1:
        dist.destroy_process_group()

    logger.info("Training completed !")
    logger.info("Model files are saved into: {0}".format(args.output_path))

    # SGLang shutdown kills loky worker processes, which crashes
    # multiprocessing.resource_tracker and leaves stray children blocking normal
    # Python exit. By this point NCCL is destroyed, SGLang is shut down and the model
    # is saved, so call os._exit to skip the child-process cleanup during
    # interpreter shutdown.
    os._exit(0)




if __name__ == '__main__':
    RANK, LOCAL_RANK, WORLD_SIZE = init_ddp() or (0,0,1)
    if not is_dist_avail_and_initialized():
        raise Exception('''DDP is not initialized; run with one of:
        python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py $args
        RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 python train.py $args
        ''')

    parse = HfArgumentParser((
        TrainArguments
    ))

    _args = parse.parse_args_into_dataclasses()[0]
    logger.info(_args)

    dist_train_plugin_guard_ind(_args)
