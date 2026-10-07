#!/usr/bin/env bash
set -euo pipefail

PYTHON_PATH=${PYTHON_PATH:-python}
PIPA_DATA_ROOT=${PIPA_DATA_ROOT:-/path/to/pipa_data_root}
CKPT_DIR=${CKPT_DIR:-./src/fusionagent/checkpoints/PIPA}
mkdir -p "${CKPT_DIR}"

"${PYTHON_PATH}" src/fusionagent/train_pipa_expert.py \
  --data-root "${PIPA_DATA_ROOT}" --cue head \
  --output "${CKPT_DIR}/pipa-head-resnet50.pth"

"${PYTHON_PATH}" src/fusionagent/train_pipa_expert.py \
  --data-root "${PIPA_DATA_ROOT}" --cue upper \
  --output "${CKPT_DIR}/pipa-upper-resnet50.pth"
