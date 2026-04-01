#!/usr/bin/env bash

set -x
export PYTHONPATH=`pwd`:$PYTHONPATH

PARTITION=$1
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
SRUN_ARGS=${SRUN_ARGS:-""}

DATASET=${2:-nuscenes}
ROOT_PATH=${3:-./data/nuscenes}
OUT_DIR=${4:-./data/nuscenes}
EXTRA_TAG=${5:-nuscenes}

srun -p ${PARTITION} \
    --job-name=create_data \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/create_data.py ${DATASET} \
            --root-path ${ROOT_PATH} \
            --out-dir ${OUT_DIR} \
            --extra-tag ${EXTRA_TAG}
