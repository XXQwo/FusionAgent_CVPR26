#!/usr/bin/env bash
set -euo pipefail

ACCELERATE_PATH=${ACCELERATE_PATH:-accelerate}
GPU_IDS=${GPU_IDS:-0,1,2,3}
NUM_GPUS=${NUM_GPUS:-4}
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export WANDB_INIT_ON_PRIMARY_PROCESS_ONLY=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${ACCELERATE_PATH}" launch \
  --gpu_ids "${GPU_IDS}" \
  --num_processes="${NUM_GPUS}" \
  --main_process_port 29543 \
  src/fusionagent/fusionagent_grpo.py \
  --configs src/fusionagent/configs/train_config_test_pipa.yaml \
  --use_accelerate
