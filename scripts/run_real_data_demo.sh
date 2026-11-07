#!/usr/bin/env bash
set -euo pipefail

DATA_FILE="${1:-examples/data_examples.jsonl}"

python -m tmu.cli \
  --mode advanced \
  --data "$DATA_FILE" \
  --epochs 1 \
  --batch-size 2 \
  --seq-len 32 \
  --max-gen-tokens 16