from statistics import mean, median, stdev

import torch
import torch.distributed as dist

cuda_time_ms = {
    'alltoall': [],
    'allreduce': [],
}

def comm_wrapper(input, op='alltoall'):
    #torch.cuda.synchronize()
    cuda_start = torch.cuda.Event(enable_timing=True)
    cuda_end = torch.cuda.Event(enable_timing=True)
    #cuda_start.record()
    input = input.contiguous()
    output = torch.empty_like(input)
    cuda_start.record()
    if op == 'alltoall':
        dist.all_to_all_single(output, input)
    else:
        dist.all_reduce(input)
    cuda_end.record()
    torch.cuda.synchronize()
    cuda_time_ms[op].append(cuda_start.elapsed_time(cuda_end))
    return output

dist.init_process_group(backend='nccl', init_method='env://')
if dist.is_initialized():
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    for step in range(6 * 10):
        input = torch.empty([4 * 1024 * 1024 * 1024], dtype=torch.int8).cuda(non_blocking=True)
        output = comm_wrapper(input, 'alltoall')
        comm_wrapper(output, 'allreduce')
        if rank == 0:
            print(step, cuda_time_ms['alltoall'][-1], cuda_time_ms['allreduce'][-1])
    if rank == 0:
        for op in cuda_time_ms:
            cuda_time_ms[op] = cuda_time_ms[op][10:]
            print(
                op,
                mean(cuda_time_ms[op]),
                median(cuda_time_ms[op]),
                max(cuda_time_ms[op]),
                min(cuda_time_ms[op]),
                stdev(cuda_time_ms[op]),
            )
