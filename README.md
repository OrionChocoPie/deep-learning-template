# Deep Learning Template
## Options to use Makefile
```bash
make  # stop, build, run

# do the same
make stop
make build
make run

make  # by default all GPUs passed 
make GPUS=all  # do the same
make GPUS=none  # without GPUs

make run GPUS=2  # pass the first two gpus
make run GPUS='\"device=0,1\"'  # pass GPUs numbered 0 and 1

make logs
make exec  # runs a new bash terminal in a running container
make exec COMMAND="bash"  # do the same 
make stop
```