#!/usr/bin/env bash
set -euo pipefail

PYTHON_PATH=${PYTHON_PATH:-python}
PIPA_META=${PIPA_META:-/path/to/PIPA_dataset}
PIPA_IMAGES=${PIPA_IMAGES:-/path/to/pipa/images}
PIPA_DATA_ROOT=${PIPA_DATA_ROOT:-/path/to/pipa_data_root}
PIPA_SPLIT=${PIPA_SPLIT:-original}

"${PYTHON_PATH}" src/fusionagent/prepare_pipa.py \
  --all-data "${PIPA_META}/all_data.txt" \
  --split-test "${PIPA_META}/split_test_${PIPA_SPLIT}.txt" \
  --images "${PIPA_IMAGES}" \
  --output "${PIPA_DATA_ROOT}"
