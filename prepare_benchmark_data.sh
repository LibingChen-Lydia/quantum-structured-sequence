#!/usr/bin/env bash
set -euo pipefail

mkdir -p dataset/ETT-small dataset/weather dataset/traffic
base='https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main'
curl -fL --retry 3 "$base/ETT-small/ETTh1.csv" -o dataset/ETT-small/ETTh1.csv
curl -fL --retry 3 "$base/ETT-small/ETTm1.csv" -o dataset/ETT-small/ETTm1.csv
curl -fL --retry 3 "$base/weather/weather.csv" -o dataset/weather/weather.csv
curl -fL --retry 3 "$base/traffic/traffic.csv" -o dataset/traffic/traffic.csv
echo 'Downloaded ETTh1, ETTm1, weather, and traffic into ./dataset.'
