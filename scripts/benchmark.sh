#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR=${1:-models/distilbert-run}

python -m src.training --output-dir "$CHECKPOINT_DIR"
python -m src.evaluate --checkpoint-path "$CHECKPOINT_DIR"
