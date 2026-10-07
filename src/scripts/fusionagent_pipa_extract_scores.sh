#!/usr/bin/env bash
set -euo pipefail

PYTHON_PATH=${PYTHON_PATH:-python}
PIPA_DATA_ROOT=${PIPA_DATA_ROOT:-/path/to/pipa_data_root}
MODES=pipa-head,pipa-upper
PIPA_PROTOCOL=${PIPA_PROTOCOL:-0to1}

# Extract expert features for the generic PIPA train identities.
for MODE in pipa-head pipa-upper; do
  "${PYTHON_PATH}" src/fusionagent/extract_features.py \
    --mode "${MODE}" --dataset pipa --root "${PIPA_DATA_ROOT}" \
    --dataset_type train --eval_mode feat
done

"${PYTHON_PATH}" src/fusionagent/extract_features.py \
  --mode "${MODES}" --dataset pipa --root "${PIPA_DATA_ROOT}" \
  --dataset_type train --eval_mode gather

# Test protocol is controlled by pipa_protocol in train_config_test_pipa.yaml.
for MODE in pipa-head pipa-upper; do
  "${PYTHON_PATH}" src/fusionagent/extract_features.py \
    --mode "${MODE}" --dataset pipa --root "${PIPA_DATA_ROOT}" \
    --dataset_type test --pipa_protocol "${PIPA_PROTOCOL}" --eval_mode feat
done

"${PYTHON_PATH}" src/fusionagent/extract_features.py \
  --mode "${MODES}" --dataset pipa --root "${PIPA_DATA_ROOT}" \
  --dataset_type test --pipa_protocol "${PIPA_PROTOCOL}" --eval_mode gather
