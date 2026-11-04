#!/usr/bin/env bash
# Full pipeline: prepare data, train both backbones, export and evaluate each.
# Usage: ./run_all.sh [epochs]   (logs to logs/<step>.log)
set -euo pipefail
cd "$(dirname "$0")"
EPOCHS="${1:-12}"
mkdir -p logs

uv run python prepare.py 2>&1 | tee logs/prepare.log
for model in dinov3 cnv2; do
  uv run python train.py --model "$model" --epochs "$EPOCHS" 2>&1 | grep --line-buffered -v Warning | tee "logs/train-$model.log"
  uv run python export.py --model "$model" 2>&1 | tee "logs/export-$model.log"
  uv run python evaluate.py --model "$model" --variant f16w 2>&1 | grep -v infer | tee "logs/evaluate-$model.log"
done
echo "PIPELINE DONE"
