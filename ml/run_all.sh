#!/usr/bin/env bash
# Full pipeline: prepare data, then train, export, evaluate and externally
# validate each model.
# Usage: ./run_all.sh [epochs] [models]   e.g. ./run_all.sh 8 dinov3
# Logs to logs/<step>.log; pause and resume with ./pause.sh.
set -euo pipefail
cd "$(dirname "$0")"
EPOCHS="${1:-12}"
MODELS="${2:-dinov3 cnv2}"
mkdir -p logs

uv run python prepare.py 2>&1 | tee logs/prepare.log
for model in $MODELS; do
  uv run python train.py --model "$model" --epochs "$EPOCHS" 2>&1 | grep --line-buffered -v Warning | tee "logs/train-$model.log"
  uv run python export.py --model "$model" 2>&1 | tee "logs/export-$model.log"
  uv run python evaluate.py --model "$model" --variant f16w 2>&1 | grep -v infer | tee "logs/evaluate-$model.log"
  if [[ -d data/ddr/DDR-dataset ]]; then
    uv run python external.py --model "$model" --variant f16w 2>&1 | grep -v infer | tee "logs/external-$model.log"
  fi
done
echo "PIPELINE DONE"
