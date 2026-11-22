#!/usr/bin/env bash
# Run from the C1 repository root in Google Cloud Shell or a GCP Linux VM.
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CPU-only PyTorch keeps this setup independent of CUDA driver versions.
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install \
  numpy==1.26.4 pandas==2.2.3 scipy==1.15.3 scikit-learn==1.6.1 \
  matplotlib==3.9.4 einops==0.8.1 reformer-pytorch==1.4.4 \
  pennylane==0.40.0

if [[ ! -d Time-Series-Library/.git ]]; then
  git clone https://github.com/thuml/Time-Series-Library.git Time-Series-Library
fi
git -C Time-Series-Library fetch --depth 1 origin 4e938a1767106324dd753b2a44832bf870a0252e
git -C Time-Series-Library checkout --detach 4e938a1767106324dd753b2a44832bf870a0252e

bash prepare_benchmark_data.sh
echo 'Environment and datasets are ready. Run: bash run_full_benchmark.sh'
