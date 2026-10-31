import os
# On ROCm, capturing hidden_states for large prefill batches allocates tens of GB
# transiently; expandable_segments mitigates OOM from reserved-but-unallocated
# fragmentation (must be set before importing torch)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import logging
import argparse
import itertools
import torch.distributed as dist

from tqdm import tqdm

from src.utils.ddp_utils import init_ddp
from src.inference import PluginGuardIndMultiTaskInference


logging.basicConfig(format='%(asctime)s -[PlugGuard]-[%(name)s]-[%(levelname)s] - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("infer")



class IterableJsonlReader:
    '''Rank-sharded JSONL reader with first-row column detection.'''
    def __init__(self, path, slice_id=0, slice_count=1):
        self.path = path
        with open(path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
        if not first_line.strip():
            raise Exception(f'Input file is empty: {path}')
        self.input_columns = list(json.loads(first_line).keys())
        with open(path, 'r', encoding='utf-8') as f:
            self.total_row_count = sum(1 for _ in f)
        self.start_pos, self.end_pos = self._get_slice_range(self.total_row_count, slice_id, slice_count)
        self.row_count = self.end_pos - self.start_pos
        print("jsonl total_row_count:{}, slice:[{},{}), row_count:{}, columns:{}".format(
            self.total_row_count, self.start_pos, self.end_pos, self.row_count, self.input_columns))

    def _get_slice_range(self, row_count, slice_id, slice_count):
        # div-mod split, each slice data count max diff 1
        size = int(row_count / slice_count)
        split_point = row_count % slice_count
        if slice_id < split_point:
            start = slice_id * (size + 1)
            end = start + (size + 1)
        else:
            start = split_point * (size + 1) + (slice_id - split_point) * size
            end = start + size
        return start, end

    def __len__(self):
        return self.row_count

    def __iter__(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in itertools.islice(f, self.start_pos, self.end_pos):
                yield json.loads(line)


def load_model(args):
    model = PluginGuardIndMultiTaskInference(
        feature_classifier_path=args.model_name_or_path,
        use_wrap_model=args.use_wrap_model,
        max_length=args.max_length,
        feature_layers=args.feature_layers,
        local_rank=LOCAL_RANK,
        stream_agg=args.stream_agg,
        llm_backend=args.llm_backend,
        tp_size=args.tp_size,
        selected_layers=args.selected_layers,
        chunked_prefill_size=args.chunked_prefill_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        attention_backend=args.attention_backend,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
    )

    return model

def run_ind_infer(args):

    if args.stream_agg != 'splitinf':
        raise ValueError(f'Unsupported --stream_agg: {args.stream_agg}, only splitinf is supported')

    model = load_model(args)

    # Input jsonl
    table_view = IterableJsonlReader(args.input_table, slice_id=RANK, slice_count=WORLD_SIZE)
    assert 'traceid' in table_view.input_columns
    if args.input_col:
        input_col = args.input_col
    else:
        input_col = 'input_text'
    assert input_col in table_view.input_columns
    # VL-only image column (optional)
    image_col = args.image_col
    if image_col:
        assert image_col in table_view.input_columns, f"image_col={image_col} not in input_columns"

    # Output jsonl: each rank writes its own shard file {output_table}.r{RANK} when WORLD_SIZE > 1
    if args.output_table is None:
        raise Exception('--output_table is required (local jsonl output path)')
    output_columns = ['traceid', 'result']
    output_path = args.output_table if WORLD_SIZE == 1 else f'{args.output_table}.r{RANK}'

    row_count = len(table_view)

    # Inference!
    write_batch_size = 10000  # flush results to file every N records
    record_list = []
    messages_list = []
    image_list = []
    output_list = []
    for i, record in enumerate(tqdm(table_view, total=row_count, mininterval=10)):

        record_list.append(record)
        messages_list.append(record[input_col])
        if image_col:
            image_list.append(record[image_col])

        if len(messages_list) < args.batch_size and i < row_count - 1:
            continue

        try:
            result_list, failed_indices = model.predict(
                messages_list,
                image_list=image_list if image_col else None,
                return_extend=args.return_extend,
                return_gate=args.return_gate,
            )
        except RuntimeError as e:
            if "chunked prefill" in str(e) or "hidden_states length mismatch with input" in str(e):
                print(f"[RANK={RANK}] WARNING: SGLang hidden-state alignment failed; check prefill token budgets and VL preprocessing; skipping this batch "
                      f"({len(messages_list)} records): {e}", flush=True)
            else:
                print(f"[RANK={RANK}] WARNING: batch inference failed, skipping this batch "
                      f"({len(messages_list)} records, traceid={[r['traceid'] for r in record_list]}): {e}", flush=True)
            record_list = []
            messages_list = []
            image_list = []
            continue

        # Skip samples whose image failed to process; do not write them to the output
        if failed_indices:
            print(f"[RANK={RANK}] skipping {len(failed_indices)} samples with failed image processing, not written to output: "
                  f"traceid={[record_list[j]['traceid'] for j in failed_indices]}")
            failed_set = set(failed_indices)
        else:
            failed_set = set()

        for idx, (record, messages, result) in enumerate(zip(record_list, messages_list, result_list)):
            if idx in failed_set:
                continue
            result = {
                'traceid': record['traceid'],
                'result': json.dumps(result, ensure_ascii=False, separators=(',', ':')),
            }
            output = [result[c] for c in output_columns]
            output_list.append(output)
        record_list = []
        messages_list = []
        image_list = []

        # Flush and display once enough results are buffered
        if len(output_list) >= write_batch_size // WORLD_SIZE or i == row_count - 1:
            if output_list:
                with open(output_path, 'a', encoding='utf-8') as f:
                    for row in output_list:
                        f.write(json.dumps({c: row[j] for j, c in enumerate(output_columns)}, ensure_ascii=False) + '\n')
                output_list = []


    if output_list:
        with open(output_path, 'a', encoding='utf-8') as f:
            for row in output_list:
                f.write(json.dumps({c: row[j] for j, c in enumerate(output_columns)}, ensure_ascii=False) + '\n')

    if WORLD_SIZE > 1:
        dist.barrier()

    # Clean up SGLang engine background threads to avoid hanging at process exit
    if model.llm_backend == "sglang":
        try:
            model.model.feature_extractor.shutdown()
        except Exception as e:
            print(f'[RANK={RANK}] SGLang shutdown warning: {e}', flush=True)

    # Tear down the NCCL process group to terminate the HeartbeatMonitor thread
    if WORLD_SIZE > 1:
        dist.destroy_process_group()

    # SGLang shutdown kills loky worker processes, which crashes
    # multiprocessing.resource_tracker and leaves stray children blocking normal
    # Python exit. By this point NCCL is destroyed, SGLang is shut down and all
    # results are written, so call os._exit to skip the child-process cleanup
    # during interpreter shutdown.
    os._exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True, help='PlugGuard classifier directory or HF repo; the backbone path comes from its config')
    parser.add_argument('--input_table', type=str, required=True, help='Local JSONL input; each row needs traceid and the input column')
    parser.add_argument('--output_table', type=str, default=None, help='Required local JSONL output; appends rows, with .r{RANK} shards for multiple ranks')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument("--use_wrap_model", default='true', type=str, help='Only true is supported')
    parser.add_argument("--stream_agg", default='splitinf', type=str, help='Streaming inference scheme; only splitinf is supported')
    parser.add_argument("--max_length", default=None, type=int, help='Required maximum input sequence length')
    parser.add_argument("--feature_layers", default=None, type=str, help='"all" or comma-separated indices into captured representations; defaults to the classifier config')
    parser.add_argument("--input_col", default=None, type=str, help='Input messages column; defaults to input_text')
    parser.add_argument("--image_col", default=None, type=str, help="Image column for VL models; leave empty for text-only models")
    parser.add_argument("--max_pixels", default=448*448, type=int, help='Maximum image area in pixels')
    parser.add_argument("--min_pixels", default=224*224, type=int, help='Minimum image area in pixels')
    parser.add_argument("--return_extend", action='store_true', help='Include marked inputs, marker masks, and per-role results')
    parser.add_argument("--return_gate", action='store_true', help='Include gate scores when available; requires --return_extend')
    parser.add_argument("--chunked_prefill_size", default=None, type=int, help="SGLang prefill token budget; None keeps the engine default and -1 disables chunking")
    parser.add_argument("--gpu_memory_utilization", default=None, type=float, help="SGLang GPU memory utilization")
    parser.add_argument("--llm_backend", default=None, type=str, choices=["hf", "sglang", "hfpp"], help='Defaults to the classifier config')
    parser.add_argument("--tp_size", default=1, type=int, help='SGLang tensor-parallel degree')
    parser.add_argument("--selected_layers", default=None, type=str, help='Absolute backbone layers captured by SGLang; defaults to the classifier config; distinct from --feature_layers')
    parser.add_argument("--attention_backend", default=None, type=str, help="SGLang attention backend; None lets SGLang choose. AMD also accepts triton_noaiter and aiter_lowmem")

    args = parser.parse_args()

    args.use_wrap_model = str(args.use_wrap_model).lower() == 'true'

    RANK, LOCAL_RANK, WORLD_SIZE = init_ddp() or (0,0,1)

    run_ind_infer(args)
