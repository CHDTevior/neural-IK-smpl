#!/usr/bin/env bash
#SBATCH --job-name=smplik_cache
#SBATCH --partition=mi300x
#SBATCH --account=ecs
#SBATCH --qos=ecsgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=800G
#SBATCH --time=2-00:00:00
#SBATCH --output=/iridisfs/scratch/ts1v23/workspace/learned_IK/smpl-ik/run_reports/cache_highmem_slurm_%j.log
#SBATCH --error=/iridisfs/scratch/ts1v23/workspace/learned_IK/smpl-ik/run_reports/cache_highmem_slurm_%j.log

set -euo pipefail

REPO_DIR="/iridisfs/scratch/ts1v23/workspace/learned_IK/smpl-ik"
ENV_DIR="/iridisfs/scratch/ts1v23/workspace/learned_IK/.envs/smplik-py38"
PYTHON="$ENV_DIR/bin/python"
DATASET_DIR="$REPO_DIR/datasets/amass_gender_augment_cache_v1"
JOB_ID="${SLURM_JOB_ID:?This script must run as a Slurm job}"
SUCCESS_FILE="$REPO_DIR/run_reports/cache_highmem_${JOB_ID}.success"
FAILURE_FILE="$REPO_DIR/run_reports/cache_highmem_${JOB_ID}.failed"

cd "$REPO_DIR"
if [[ ! -x "$PYTHON" ]]; then
  echo "Pinned SMPL-IK Python is unavailable: $PYTHON" >&2
  exit 1
fi
if [[ -e "$SUCCESS_FILE" ]] || [[ -e "$FAILURE_FILE" ]]; then
  echo "A high-memory cache sentinel already exists; refusing to overwrite" >&2
  exit 1
fi
if find "$DATASET_DIR" -maxdepth 1 -name '*_cache.feather' -print -quit |
  grep -q .; then
  echo "A cache file already exists; refusing an ambiguous rebuild" >&2
  exit 1
fi
for required_file in \
  "$DATASET_DIR/dataset_settings.json" \
  "$DATASET_DIR/split.json" \
  "$DATASET_DIR/preprocess_manifest.json"; do
  if [[ ! -s "$required_file" ]]; then
    echo "Missing completed preprocessing artifact: $required_file" >&2
    exit 1
  fi
done

record_failure() {
  local status="$?"
  printf 'exit_status=%s time=%s job_id=%s\n' \
    "$status" "$(date --iso-8601=seconds)" "${SLURM_JOB_ID:-unknown}" \
    >"$FAILURE_FILE"
  exit "$status"
}
trap record_failure ERR

echo "Start time: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Job ID: $JOB_ID"
echo "Cgroup: $(cut -d: -f3 /proc/self/cgroup)"
free -h

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
"$PYTHON" ops/build_dataset_cache.py

printf 'time=%s job_id=%s\n' \
  "$(date --iso-8601=seconds)" "$JOB_ID" >"$SUCCESS_FILE"
trap - ERR
