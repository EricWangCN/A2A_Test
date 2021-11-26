# Command to test:
```shell
NCCL_DEBUG=INFO OMP_NUM_THREADS=1 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 <*.py>
```
# Profile with NSight Systems:
NVIDIA NSight Systems can be downloaded [here](https://developer.nvidia.com/gameworksdownload) (NVIDIA Developer account is needed). 

```shell
$ nsys profile --trace=cuda --cuda-memory-usage=true --sample=none --export=sqlite --show-output=true --force-overwrite=true --output=<path> bash run.sh cudaeventtime
$ nsys profile --trace=cuda --cuda-memory-usage=true --sample=none --export=sqlite --show-output=true --force-overwrite=true --output=<path> bash run.sh cputime
```
