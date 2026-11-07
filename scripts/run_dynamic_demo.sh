#!/usr/bin/env bash
set -euo pipefail

python -m tmu.cli \
  --mode advanced \
  --epochs 1 \
  --batch-size 2 \
  --seq-len 32 \
  --samples 64 \
  --dynamic \
  --per-layer-timing \
  --use-gpu-events \
  --max-gen-tokens 12