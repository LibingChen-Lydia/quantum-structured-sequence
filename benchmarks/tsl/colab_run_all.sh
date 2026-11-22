#!/usr/bin/env bash
# One-shot Colab launcher: everything runs in the background, logs in ./logs/.
#   bash benchmarks/tsl/colab_run_all.sh [MAX_HOURS]
# GPU: 6 workers share one job queue for ETTh1/ETTm1/weather (seeds 2021-2023, longest first),
#      2 workers for Traffic Transformer/iTransformer (seed 2021), 1 for Traffic PatchTST H=96.
# CPU: Module C on Traffic (seed 2021, memory stride 2).
set -euo pipefail
cd "$(dirname "$0")/../.."
H=${1:-1.6}
mkdir -p logs
RB="python benchmarks/tsl/run_baselines.py --device cuda --num-workers 1 --max-hours $H"
python benchmarks/tsl/run_baselines.py --clear-locks
for i in 0 1 2 3 4 5; do
  nohup $RB --datasets ETTh1 ETTm1 weather > logs/gpu$i.log 2>&1 &
done
for i in 0 1; do
  nohup $RB --datasets traffic --seeds 2021 --models Transformer iTransformer > logs/traffic$i.log 2>&1 &
done
nohup $RB --datasets traffic --seeds 2021 --models PatchTST --horizons 96 > logs/traffic_patchtst.log 2>&1 &
nohup sh -c 'for h in 96 192 336 720; do PYTHONPATH=src OMP_NUM_THREADS=2 python experiments/retrieval/run_module_c_tsl.py --dataset traffic --pred-len $h --seed 2021 --memory-stride 2; done' > logs/module_c_traffic.log 2>&1 &
sleep 2
echo "launched: $(pgrep -fc run_baselines.py) baseline workers, Module C traffic loop; no new jobs after ${H} h"
