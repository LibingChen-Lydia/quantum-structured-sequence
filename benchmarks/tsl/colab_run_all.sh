#!/usr/bin/env bash
# One-shot Colab launcher (A100): ETTh1 + Weather, seeds 2021-2023. Logs in ./logs/.
#   bash benchmarks/tsl/colab_run_all.sh [MAX_HOURS]
# GPU: 8 workers share one job queue (3 models x 4 horizons x 3 seeds x 2 datasets = 72 jobs,
#      seed-first, longest-first). CPU: Module C, 3 parallel workers (24 jobs).
set -euo pipefail
cd "$(dirname "$0")/../.."
H=${1:-1.2}
mkdir -p logs
python benchmarks/tsl/run_baselines.py --clear-locks
for i in 0 1 2 3 4 5 6 7; do
  nohup python benchmarks/tsl/run_baselines.py --device cuda --num-workers 1 --max-hours "$H" \
    --datasets ETTh1 weather --seeds 2021 2022 2023 > "logs/gpu$i.log" 2>&1 &
done
PYTHON=python nohup bash benchmarks/tsl/run_module_c_queue.sh 3 "ETTh1 weather" > logs/module_c.log 2>&1 &
sleep 2
echo "launched $(pgrep -fc run_baselines.py) GPU workers + Module C queue; no new GPU job after ${H} h"
