# Command to test:
```shell
NCCL_DEBUG=INFO OMP_NUM_THREADS=1 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 main.py
```
# Profile with NSight Systems:
```shell
nsys profile --trace=cuda --cuda-memory-usage=true --sample=none --export=sqlite --show-output=true --force-overwrite=true --output=<path> bash run.sh
```
