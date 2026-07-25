#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${SMPLIK_PYTHON:-/iridisfs/scratch/ts1v23/workspace/learned_IK/.envs/smplik-py38/bin/python}"
RAW_ROOT="${AMASS_RAW_ROOT:-/iridisfs/scratch/ts1v23/workspace/motion-latent-diffusion-main/datasets/amass/motion_data}"
OUTPUT_ROOT="$REPO_DIR/datasets/amass_gender_augment_cache_v1"
REPORT_ROOT="$REPO_DIR/run_reports/amass_preprocess"

mkdir -p "$OUTPUT_ROOT" "$REPORT_ROOT"
cd "$REPO_DIR"

launch_worker() {
  local gpu="$1"
  local worker="$2"
  shift 2
  CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" \
    ops/preprocess_amass.py \
    --source-root "$RAW_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --report-root "$REPORT_ROOT" \
    convert \
    --device cuda \
    "$@" >"$REPORT_ROOT/worker_${worker}.log" 2>&1 &
  WORKER_PIDS+=("$!")
}

WORKER_PIDS=()
launch_worker 0 0 KIT_smplh:KIT MPI_Limits TotalCapture TCD_handMocap SSM_synced
launch_worker 1 1 BioMotionLab_NTroje BMLhandball
launch_worker 2 2 CMU BMLmovi
launch_worker 3 3 Eyes_Japan_Dataset MPI_HDM05 ACCAD EKUT \
  Transitions_mocap MPI_mosh SFU HumanEva DFaust_67

worker_status=0
for worker_pid in "${WORKER_PIDS[@]}"; do
  if ! wait "$worker_pid"; then
    worker_status=1
  fi
done

if [[ "$worker_status" -ne 0 ]]; then
  echo "At least one AMASS preprocessing worker failed; see $REPORT_ROOT/worker_*.log" >&2
  exit "$worker_status"
fi

PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" ops/preprocess_amass.py \
  --source-root "$RAW_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --report-root "$REPORT_ROOT" \
  finalize | tee "$REPORT_ROOT/finalize.log"
