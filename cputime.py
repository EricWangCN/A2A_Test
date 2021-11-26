from statistics import mean, median, stdev

import torch
import torch.distributed as dist

cuda_time_ms = {
    'alltoall': [],
    'allreduce': [],
}

def comm_wrapper(input, op='alltoall'):
    #torch.cuda.synchronize()
    #cuda_start.record()
    input = input.contiguous()
    import time
    torch.cuda.synchronize()
    start = time.time()
    output = torch.empty_like(input)
    if op == 'alltoall':
        dist.all_to_all_single(output, input)
    else:
        dist.all_reduce(input)
    torch.cuda.synchronize()
    end = time.time()
    return output, 1000 * (end - start)

dist.init_process_group(backend='nccl', init_method='env://')
if dist.is_initialized():
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    input = torch.empty([4 * 1024 * 1024 * 1024], dtype=torch.int8).cuda(non_blocking=True)
    for step in range(6 * 10):
        output, a2a_cpu_time_sync = comm_wrapper(input, 'alltoall')
        output, allreduce_cpu_time_sync = comm_wrapper(output, 'allreduce')
        if rank == 0:
            print('rank', rank, step, a2a_cpu_time_sync, allreduce_cpu_time_sync)
