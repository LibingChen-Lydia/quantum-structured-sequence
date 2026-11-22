# C1 full forecasting benchmark

This branch evaluates `Transformer`, `iTransformer`, `PatchTST`, and `DLinear`
from THUML's Time-Series-Library plus the project `QSS_HierarchicalRetriever`
on ETTh1, ETTm1, Weather, and Traffic. Every model uses an input length of 96
and prediction lengths 96, 192, 336, and 420. Test windows slide one time step
at a time. The target is the final numeric CSV column (OT on ETTh1/ETTm1).

TSL uses its standard split: the ETT sets use 12/4/4 months; Weather and Traffic
use chronological 70/10/20 splits. Scaling is fit on train data only. Baselines
run in TSL's supported multivariate-to-single-target (`MS`) mode. MSE and MAE in
`results/final_metrics.csv` are on train-standardized target values; QSS rows
also include `mae_raw` and `mse_raw` in the target's original units.

## Run from GCP Cloud Shell

Upload this repository folder to Cloud Shell, then run:

```bash
bash setup_gcp.sh
bash run_full_benchmark.sh
```

`setup_gcp.sh` installs CPU-only PyTorch and the dependencies for these four
models, checks out Time-Series-Library commit
`4e938a1767106324dd753b2a44832bf870a0252e`, and downloads the four datasets
from the THUML Hugging Face dataset. The long-running benchmark can be resumed:
the baseline runner skips successful rows already written to
`results/baseline_metrics.csv`; model logs are under `results/logs/`.

Cloud Shell is free but has a 50-hour weekly quota, sessions capped at 12 hours,
and 5 GB persistent storage. This full suite may exceed Cloud Shell's time or
disk limits; if so, use the same scripts on a Compute Engine VM. The scripts do
not create or start a VM.
