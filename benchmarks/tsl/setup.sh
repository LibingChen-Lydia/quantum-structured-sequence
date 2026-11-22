#!/usr/bin/env bash
# Prepare Time-Series-Library (pinned), its datasets and the seed patch. Works on macOS and Colab.
#   bash benchmarks/tsl/setup.sh            # uses the current python / pip
set -euo pipefail
cd "$(dirname "$0")/../.."
TSL=third_party/Time-Series-Library
COMMIT=4e938a1767106324dd753b2a44832bf870a0252e
PY=${PYTHON:-python}

if [[ ! -d $TSL/.git ]]; then
  git clone https://github.com/thuml/Time-Series-Library.git "$TSL"
fi
git -C "$TSL" fetch -q origin "$COMMIT" || true
git -C "$TSL" checkout -q --detach "$COMMIT"

# Only the packages the long-term-forecast path needs (TSL's full requirements.txt also pulls
# foundation-model stacks such as timesfm/jax). Versions follow TSL's requirements.txt.
# torch is NOT installed here: Colab ships a CUDA build, locally the project venv has one.
$PY -m pip install -q einops==0.8.1 sktime==0.40.1 datasets==4.5.0 huggingface_hub==0.36.0 \
  PyWavelets==1.9.0 patool==4.0.3 reformer-pytorch==1.4.4 local-attention==1.11.2 tqdm "pandas>=2.2,<3" scikit-learn pennylane pyyaml

# Seed patch: TSL hard-codes fix_seed = 2021 in run.py; read it from TSL_SEED instead
# (default 2021, so unpatched behaviour is unchanged).
if grep -q "fix_seed = 2021" "$TSL/run.py"; then
  sed -i.bak 's/    fix_seed = 2021/    fix_seed = int(os.environ.get("TSL_SEED", 2021))/' "$TSL/run.py"
fi
# MPS fix: run.py calls torch.backends.mps.empty_cache(), which does not exist (torch.mps does).
sed -i.bak 's/torch\.backends\.mps\.empty_cache()/torch.mps.empty_cache()/' "$TSL/run.py"
grep -n "fix_seed = " "$TSL/run.py"

# Datasets from TSL's official Hugging Face mirror (README: thuml/Time-Series-Library)
base=https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main
mkdir -p "$TSL/dataset/ETT-small" "$TSL/dataset/weather"
for f in ${TSL_DATA:-ETT-small/ETTh1.csv weather/weather.csv}; do
  [[ -s $TSL/dataset/$f ]] || curl -fL --retry 3 -s "$base/$f" -o "$TSL/dataset/$f"
done
ls -la "$TSL"/dataset/*/
echo "TSL ready at $TSL"
