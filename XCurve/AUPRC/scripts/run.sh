#!/bin/sh
config=$1
gpu_id=$2
export CUDA_VISIBLE_DEVICES=${gpu_id}
python train.py ${config}
# python eval.py ${config}
