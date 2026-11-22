#!/usr/bin/env bash
# Module C over datasets x horizons x seeds, seed-first, N parallel CPU workers. Resumable.
#   bash benchmarks/tsl/run_module_c_queue.sh 4 "ETTh1 ETTm1 weather" [extra args...]
set -euo pipefail
cd "$(dirname "$0")/../.."
N=${1:-4}; DATASETS=${2:-"ETTh1 ETTm1 weather"}; shift 2 || true
PY=${PYTHON:-python}
export PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p outputs/benchmark/module_c/logs
for s in 2021 2022 2023 2024 2025; do for d in $DATASETS; do for h in 96 192 336 720; do
  echo "$d $h $s"
done; done; done | xargs -P "$N" -L 1 sh -c \
  "$PY experiments/retrieval/run_module_c_tsl.py --dataset \$0 --pred-len \$1 --seed \$2 $* \
   > outputs/benchmark/module_c/logs/\$0_\$1_s\$2.log 2>&1 && echo done \$0 \$1 \$2 || echo FAILED \$0 \$1 \$2"
