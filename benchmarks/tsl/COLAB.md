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
- downloads ETTh1 and Weather from TSL's Hugging Face mirror.

It keeps Colab's own CUDA torch and does not reinstall it.

**2. Launch everything in the background (one cell)**

```bash
%cd /content/qss
!bash benchmarks/tsl/colab_run_all.sh
```

- **Scope.** ETTh1 and Weather, seeds 2021, 2022 and 2023.
- **GPU.** 8 workers share one queue of 72 jobs (Transformer, iTransformer and PatchTST × 4
  horizons × 3 seeds × 2 datasets). Each worker claims the next free job with a lock file. Seed
  2021 finishes before 2022 starts, and within a seed the longest jobs run first.
- **CPU.** Module C runs 24 jobs (2 datasets × 4 horizons × 3 seeds) on 3 parallel workers.
- **Time limit.** No new GPU job starts after 1.2 hours; jobs that are already running finish.
  To stop immediately, run
  `!pkill -f run_baselines.py; pkill -f run.py; pkill -f run_module_c`.
- **Expected time.** About 40–60 minutes. This is an estimate, not yet measured on an A100.

**3. Check progress (re-run whenever you like)**

```bash
!tail -qn 1 /content/qss/logs/*.log; ls /content/qss/outputs/benchmark/tsl/runs/*.json | wc -l; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
```

**4. Hand back the results**

The JSON files are small. Re-run this cell at any time; it zips whatever has finished.

```python
!cd /content/qss && zip -qr tsl_runs.zip outputs/benchmark/tsl/runs/*.json outputs/benchmark/tsl/runs/logs outputs/benchmark/module_c/*.json
from google.colab import files; files.download('/content/qss/tsl_runs.zip')
```

Put `tsl_runs.zip` in `~/Downloads`. `make_report.py` merges it with the local results.

## If the session disconnects

Re-run cells 1 and 2 (the launcher clears stale locks). Every finished `(model, dataset, horizon, seed)` has a JSON with
`"status": "OK"` and is skipped, so the run continues where it stopped. Download the zip first if
you can: the Colab disk is wiped when the runtime is recycled.

## Notes

- Hyper-parameters come from TSL's `scripts/long_term_forecast/*.sh`, parsed per horizon.
- TSL has no iTransformer ETTh1 script, so its ETTh2 script is used (d_model = d_ff = 128,
  e_layers = 2). Each record stores this in `script_source`.
