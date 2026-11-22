# Run the TSL baselines on Colab (A100)

Runtime → Change runtime type → **A100 GPU** (High-RAM). Paste each block into its own cell.

**1. Get the code, TSL, data and the seed patch (about 2 minutes)**

The repository is private, so pick one way to get the code onto Colab:

- **(a) Upload a zip.** Drag `qss_benchmark.zip` (made with
  `git archive -o qss_benchmark.zip benchmark-tsl`) into the Colab file pane, then run
  `!mkdir -p /content/qss && unzip -q -o /content/qss_benchmark.zip -d /content/qss`.
- **(b) Clone with a token.** Store a GitHub token in Colab Secrets as `GH_TOKEN`.

```python
# option (b) only
from google.colab import userdata; import os; os.environ["GH_TOKEN"] = userdata.get("GH_TOKEN")
!git clone -q -b benchmark-tsl https://$GH_TOKEN@github.com/LibingChen-Lydia/quantum-structured-sequence.git /content/qss
```

```bash
%cd /content/qss
!bash benchmarks/tsl/setup.sh
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

`setup.sh` does three things:

- pins Time-Series-Library at `4e938a1`;
- patches `run.py` so `fix_seed` comes from `TSL_SEED`;
- downloads ETTh1, ETTm1, Weather and Traffic from TSL's Hugging Face mirror.

It keeps Colab's own CUDA torch and does not reinstall it.

**2. Start 4 parallel workers on the one GPU (non-blocking)**

```bash
%cd /content/qss
!mkdir -p logs
# 3 shards share ETTh1 + ETTm1 + Weather, seed-first (all cells get seed 2021 before 2022 ...)
!for i in 0 1 2; do nohup python benchmarks/tsl/run_baselines.py --device cuda --datasets ETTh1 ETTm1 weather --shard $i/3 --max-hours 1.8 > logs/shard$i.log 2>&1 & done
# 1 worker for Traffic (862 variables, slow): seed 2021, horizons 96 and 720 first
!nohup sh -c "python benchmarks/tsl/run_baselines.py --device cuda --datasets traffic --seeds 2021 --horizons 96 720 --max-hours 1.8 && python benchmarks/tsl/run_baselines.py --device cuda --datasets traffic --seeds 2021 --horizons 192 336 --max-hours 0.8" > logs/traffic.log 2>&1 &
```

**2b. Module C on Traffic (CPU, needs the High-RAM runtime)**

Traffic has 862 variables. At horizon 720 the memory bank alone is about 28 GB, so this part
runs on Colab, not on a 16 GB laptop. It runs one seed, keeps every 2nd training window in the
memory bank, and records `memory_stride` in each result.

```bash
%cd /content/qss
!nohup sh -c 'for h in 96 192 336 720; do PYTHONPATH=src OMP_NUM_THREADS=4 python experiments/retrieval/run_module_c_tsl.py --dataset traffic --pred-len $h --seed 2021 --memory-stride 2; done' > logs/module_c_traffic.log 2>&1 &
```

**3. Check progress (re-run whenever you like)**

```bash
!tail -n 3 /content/qss/logs/*.log; ls /content/qss/outputs/benchmark/tsl/runs/*.json | wc -l
```

**4. Hand back the results**

The JSON files are small. Re-run this cell at any time; it zips whatever has finished.

```python
!cd /content/qss && zip -qr tsl_runs.zip outputs/benchmark/tsl/runs/*.json outputs/benchmark/tsl/runs/logs outputs/benchmark/module_c/*.json
from google.colab import files; files.download('/content/qss/tsl_runs.zip')
```

Put `tsl_runs.zip` in `~/Downloads`. `make_report.py` merges it with the local results.

## If the session disconnects

Re-run cells 1 and 2. Every finished `(model, dataset, horizon, seed)` has a JSON with
`"status": "OK"` and is skipped, so the run continues where it stopped. Download the zip first if
you can: the Colab disk is wiped when the runtime is recycled.

## Notes

- Hyper-parameters come from TSL's `scripts/long_term_forecast/*.sh`, parsed per horizon.
- TSL has no iTransformer ETTh1/ETTm1 script, so its ETTh2 script is used (d_model = d_ff = 128,
  e_layers = 2). Each record stores this in `script_source`.
- PatchTST on Traffic uses TSL's batch size 4 and is the slowest job. If Traffic does not finish,
  it is reported as incomplete; numbers from papers are never used to fill it.
