#!/usr/bin/env bash
set -euo pipefail

PYTHON_PATH=${PYTHON_PATH:-python}
PIPA_DATA_ROOT=${PIPA_DATA_ROOT:-/path/to/pipa_data_root}
CFG=./src/fusionagent/WBModules/model_cfg_pipa.yaml

for MODE in pipa-head pipa-upper; do
  "${PYTHON_PATH}" src/fusionagent/precompute_center.py \
    --mode "${MODE}" --dataset pipa --root "${PIPA_DATA_ROOT}" \
    --backbone_cfg "${CFG}" --save_path ./src/fusionagent/mod_center_feat/
done
