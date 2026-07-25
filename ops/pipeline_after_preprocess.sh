#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${SMPLIK_ENV_DIR:-/iridisfs/scratch/ts1v23/workspace/learned_IK/.envs/smplik-py38}"
PREPROCESS_PID="${SMPLIK_PREPROCESS_PID:?Set SMPLIK_PREPROCESS_PID to the active preprocessing launcher PID}"
GPU_INDEX="${SMPLIK_GPU_INDEX:-0}"
CACHE_MODE="${SMPLIK_CACHE_MODE:?Set SMPLIK_CACHE_MODE to highmem}"
CACHE_JOB_ID="${SMPLIK_CACHE_JOB_ID:-}"
PYTHON="$ENV_DIR/bin/python"
DATASET_DIR="$REPO_DIR/datasets/amass_gender_augment_cache_v1"
FORMAL_JOB_DIR="$REPO_DIR/logs/protores_amass/seed-0_num_blocks_stage1-3"
FORMAL_LOG="$REPO_DIR/run_reports/formal_training.log"

cd "$REPO_DIR"
if [[ ! -x "$PYTHON" ]]; then
  echo "Pinned SMPL-IK Python is unavailable: $PYTHON" >&2
  exit 1
fi
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

echo "Waiting for AMASS preprocessing PID $PREPROCESS_PID"
while kill -0 "$PREPROCESS_PID" 2>/dev/null; do
  sleep 30
done

for required_file in \
  "$DATASET_DIR/dataset_settings.json" \
  "$DATASET_DIR/split.json" \
  "$DATASET_DIR/preprocess_manifest.json"; do
  if [[ ! -s "$required_file" ]]; then
    echo "Preprocessing did not produce required file: $required_file" >&2
    exit 1
  fi
done

date
if [[ "$CACHE_MODE" == "highmem" ]]; then
  if [[ -z "$CACHE_JOB_ID" ]]; then
    echo "SMPLIK_CACHE_JOB_ID is required for highmem cache mode" >&2
    exit 1
  fi
  CACHE_SUCCESS="$REPO_DIR/run_reports/cache_highmem_${CACHE_JOB_ID}.success"
  CACHE_FAILURE="$REPO_DIR/run_reports/cache_highmem_${CACHE_JOB_ID}.failed"
  echo "Waiting for high-memory cache job $CACHE_JOB_ID"
  while [[ ! -s "$CACHE_SUCCESS" ]]; do
    if [[ -s "$CACHE_FAILURE" ]]; then
      echo "High-memory cache job reported failure: $CACHE_FAILURE" >&2
      exit 1
    fi
    if ! squeue -h -j "$CACHE_JOB_ID" | grep -q .; then
      cache_state="$(
        sacct -n -X -j "$CACHE_JOB_ID" --format=State |
          awk 'NF {print $1; exit}'
      )"
      if [[ -n "$cache_state" ]] &&
         [[ "$cache_state" != PENDING* ]] &&
         [[ "$cache_state" != RUNNING* ]] &&
         [[ "$cache_state" != COMPLETING* ]]; then
        echo "High-memory cache job ended in state $cache_state" >&2
        exit 1
      fi
    fi
    sleep 30
  done
  for cache_file in \
    "$DATASET_DIR/Training_cache.feather" \
    "$DATASET_DIR/Validation_cache.feather" \
    "$DATASET_DIR/Test_cache.feather" \
    "$REPO_DIR/run_reports/dataset_cache_build.json"; do
    if [[ ! -s "$cache_file" ]]; then
      echo "High-memory cache completion is missing: $cache_file" >&2
      exit 1
    fi
  done
else
  echo "Unknown cache mode: $CACHE_MODE" >&2
  exit 1
fi

"$PYTHON" ops/dataset_audit.py 2>&1 |
  tee run_reports/dataset_audit.txt
"$PYTHON" ops/validate_batch.py 2>&1 |
  tee run_reports/batch_validation.txt

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
"$PYTHON" ops/smoke_test.py 2>&1 |
  tee run_reports/smoke_test.log

{
  date
  git status --short
  sha256sum \
    configs/experiments/smplik_amass.yaml \
    configs/posing_smpl.yaml \
    configs/model/posing_smpl_default.yaml \
    smplik/models/smpl_model.py
  git diff -- \
    configs/experiments/smplik_amass.yaml \
    configs/posing_smpl.yaml \
    configs/model/posing_smpl_default.yaml \
    smplik/models/smpl_model.py
  git diff --cached -- \
    configs/experiments/smplik_amass.yaml \
    configs/posing_smpl.yaml \
    configs/model/posing_smpl_default.yaml \
    smplik/models/smpl_model.py
} > run_reports/prelaunch_integrity.txt

if ! printf '%s\n' \
  '07621dd6d361f6db4e813ddb974f9a9364e3c1468c66cbad1c3d89ebdc537d06  configs/experiments/smplik_amass.yaml' \
  '7b7a9bc62b44e697a60e61d43e07397595d4db4be3162dfad792d857ee06b549  configs/posing_smpl.yaml' \
  'b8ac2f94b971aeca344cbdc99d42221a75e2d58682bc119f275195decb17d9eb  configs/model/posing_smpl_default.yaml' \
  'a3c894a62fb441d37f423a777566062d6dcfebc06dbe5f6da8440d8e8d0156dc  smplik/models/smpl_model.py' |
  sha256sum --check --status; then
  echo "Protected training file checksum mismatch; formal launch blocked" >&2
  exit 1
fi
if ! git diff --quiet -- \
  configs/experiments/smplik_amass.yaml \
  configs/posing_smpl.yaml \
  configs/model/posing_smpl_default.yaml \
  smplik/models/smpl_model.py; then
  echo "Protected training files changed; formal launch blocked" >&2
  exit 1
fi
if ! git diff --cached --quiet -- \
  configs/experiments/smplik_amass.yaml \
  configs/posing_smpl.yaml \
  configs/model/posing_smpl_default.yaml \
  smplik/models/smpl_model.py; then
  echo "Protected training files have staged changes; formal launch blocked" >&2
  exit 1
fi
if [[ -e "$FORMAL_JOB_DIR/hparams.yaml" ]] ||
   find "$FORMAL_JOB_DIR/checkpoints" -maxdepth 1 -name '*.ckpt' -print -quit 2>/dev/null |
     grep -q .; then
  echo "A prior formal run exists; from-scratch launch blocked" >&2
  exit 1
fi
if [[ -e "$FORMAL_LOG" ]]; then
  echo "Formal log already exists; refusing to overwrite: $FORMAL_LOG" >&2
  exit 1
fi

{
  echo "Start time: $(date --iso-8601=seconds)"
  echo "GPU index: $GPU_INDEX"
  echo "Exact command: python run.py --config=configs/experiments/smplik_amass.yaml"
} | tee run_reports/formal_training_start.txt

bash ops/train.sh >"$FORMAL_LOG" 2>&1
