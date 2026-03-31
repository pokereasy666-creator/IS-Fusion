#!/usr/bin/env bash

set -x
export PYTHONPATH=`pwd`:$PYTHONPATH

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
WORK_DIR=$4
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
SRUN_ARGS=${SRUN_ARGS:-""}
JOB_NAME=create_data

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/create_data.py ${5:-nuscenes} \
            --root-path ${6:-./data/nuscenes} \
            --out-dir ${7:-./data/nuscenes} \
            --extra-tag ${8:-nuscenes}
