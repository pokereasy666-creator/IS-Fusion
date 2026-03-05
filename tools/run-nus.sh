#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(cd "$(dirname "$0")/.." && pwd)

TASK_DESC=$1
PORT=$((8000 + RANDOM %57535))

CONFIG=configs/isfusion/isfusion_0075voxel.py

# Single A30 GPU (24GB) - full resolution, no channel reduction needed
CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch --nproc_per_node=1 --master_port ${PORT} $(dirname "$0")/train.py --launcher pytorch $CONFIG \
--extra_tag $TASK_DESC




