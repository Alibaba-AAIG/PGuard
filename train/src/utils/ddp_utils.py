import os
import pickle
import datetime
import torch
import torch.distributed as dist

def init_ddp():
    dist_backend = 'nccl'
    dist_url = 'env://'

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        # Match the launcher's per-rank visible-device layout before NCCL init.
        # Modulo also keeps the device index valid when each process sees one GPU.
        gpu = rank % torch.cuda.device_count()
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        return

    print(f'| [DDP Init] RANK={rank}, LOCAL_PROCESS_RANK={os.environ.get("LOCAL_PROCESS_RANK", "N/A")}, LOCAL_RANK={os.environ.get("LOCAL_RANK", "N/A")}, gpu={gpu}, device_count={torch.cuda.device_count()}, CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "not set")}', flush=True)

    torch.cuda.set_device(gpu)

    # NCCL deadlock diagnostics: enable FlightRecorder (dumps each rank's collective
    # sequence on timeout) and shorten the watchdog timeout from 1h to 15min
    os.environ.setdefault('TORCH_NCCL_TRACE_BUFFER_SIZE', '1000')
    os.environ.setdefault('TORCH_NCCL_DUMP_ON_TIMEOUT', '1')
    print(f'[RANK={rank}] NCCL timeout=15min, FlightRecorder enabled (dumps go to the container-local /tmp by default)', flush=True)

    dist.init_process_group(
        backend=dist_backend,
        init_method=dist_url,
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(minutes=15),
        )
    dist.barrier()
    return rank, gpu, world_size

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def all_gather(data):
    """Gather arbitrary picklable data from every rank."""
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

def dist_barrier():
    if is_dist_avail_and_initialized():
        dist.barrier()

if __name__ == "__main__":
    RANK, LOCAL_RANK, WORLD_SIZE = init_ddp() or (0,0,1)
