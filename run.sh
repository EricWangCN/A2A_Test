NCCL_DEBUG=INFO OMP_NUM_THREADS=1 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 $1.py
