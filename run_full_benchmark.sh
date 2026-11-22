#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
mkdir -p results
python benchmark_c.py --datasets ETTh1 ETTm1 weather traffic --results results
python benchmark_tsl.py --tsl-root Time-Series-Library --dataset-root dataset \
  --results results --datasets ETTh1 ETTm1 weather traffic \
  --models Transformer iTransformer PatchTST DLinear \
  --horizons 96 192 336 420 --epochs 10 --batch-size 32 --resume
python combine_metrics.py --results results
echo 'Full evaluation complete: results/final_metrics.csv'
