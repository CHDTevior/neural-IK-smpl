#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${SMPLIK_ENV_DIR:-/iridisfs/scratch/ts1v23/workspace/learned_IK/.envs/smplik-py38}"

cd "$REPO_DIR"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "Pinned SMPL-IK Python is unavailable: $ENV_DIR/bin/python" >&2
  exit 1
fi
export PATH="$ENV_DIR/bin:$PATH"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

python run.py --config=configs/experiments/smplik_amass.yaml
