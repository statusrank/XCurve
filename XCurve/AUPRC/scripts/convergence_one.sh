#!/bin/sh
config=$1
gpu_id=$2
export CUDA_VISIBLE_DEVICES=${gpu_id}
python convergence.py ${config}
