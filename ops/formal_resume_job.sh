#!/usr/bin/env bash
#SBATCH --job-name=smplik_amass_resume
#SBATCH --partition=a100
#SBATCH --account=ecs
#SBATCH --qos=ecsgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=2-12:00:00
#SBATCH --output=/iridisfs/scratch/ts1v23/workspace/learned_IK/smpl-ik/run_reports/formal_resume_slurm_%j.log
#SBATCH --error=/iridisfs/scratch/ts1v23/workspace/learned_IK/smpl-ik/run_reports/formal_resume_slurm_%j.log

set -euo pipefail

REPO_DIR="/iridisfs/scratch/ts1v23/workspace/learned_IK/smpl-ik"
ENV_DIR="/iridisfs/scratch/ts1v23/workspace/learned_IK/.envs/smplik-py38"
FORMAL_DIR="$REPO_DIR/logs/protores_amass/seed-0_num_blocks_stage1-3"
COMPLETED_MARKER="$REPO_DIR/run_reports/formal_training.completed"

cd "$REPO_DIR"
if [[ -s "$COMPLETED_MARKER" ]]; then
  echo "Formal training is already complete: $COMPLETED_MARKER"
  exit 0
fi
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "Pinned SMPL-IK Python is unavailable: $ENV_DIR/bin/python" >&2
  exit 1
fi
if [[ ! -s "$FORMAL_DIR/hparams.yaml" ]]; then
  echo "Cannot resume without the formal hparams.yaml" >&2
  exit 1
fi

latest_checkpoint="$(
  {
    find "$FORMAL_DIR/checkpoints" -maxdepth 1 -type f -name '*.ckpt' \
      -printf '%T@ %s %p\n' 2>/dev/null || true
  } |
    sort -nr |
    head -n 1
)"
if [[ -z "$latest_checkpoint" ]]; then
  echo "Cannot resume without a formal checkpoint" >&2
  exit 1
fi
checkpoint_size="$(awk '{print $2}' <<<"$latest_checkpoint")"
checkpoint_path="$(cut -d' ' -f3- <<<"$latest_checkpoint")"
if [[ "$checkpoint_size" -le 0 ]]; then
  echo "Latest formal checkpoint is empty: $checkpoint_path" >&2
  exit 1
fi

if ! printf '%s\n' \
  '07621dd6d361f6db4e813ddb974f9a9364e3c1468c66cbad1c3d89ebdc537d06  configs/experiments/smplik_amass.yaml' \
  '7b7a9bc62b44e697a60e61d43e07397595d4db4be3162dfad792d857ee06b549  configs/posing_smpl.yaml' \
  'b8ac2f94b971aeca344cbdc99d42221a75e2d58682bc119f275195decb17d9eb  configs/model/posing_smpl_default.yaml' \
  'a3c894a62fb441d37f423a777566062d6dcfebc06dbe5f6da8440d8e8d0156dc  smplik/models/smpl_model.py' |
  sha256sum --check --status; then
  echo "Protected training file checksum mismatch; resume blocked" >&2
  exit 1
fi
if ! git diff --quiet -- \
  configs/experiments/smplik_amass.yaml \
  configs/posing_smpl.yaml \
  configs/model/posing_smpl_default.yaml \
  smplik/models/smpl_model.py; then
  echo "Protected training files changed; resume blocked" >&2
  exit 1
fi
if ! git diff --cached --quiet -- \
  configs/experiments/smplik_amass.yaml \
  configs/posing_smpl.yaml \
  configs/model/posing_smpl_default.yaml \
  smplik/models/smpl_model.py; then
  echo "Protected training files have staged changes; resume blocked" >&2
  exit 1
fi

export PATH="$ENV_DIR/bin:$PATH"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

echo "Resume time: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Slurm job: ${SLURM_JOB_ID:-unknown}"
echo "Latest checkpoint: $checkpoint_path ($checkpoint_size bytes)"
echo "Exact command: python run.py --config=configs/experiments/smplik_amass.yaml"

python run.py --config=configs/experiments/smplik_amass.yaml

printf 'time=%s job_id=%s host=%s\n' \
  "$(date --iso-8601=seconds)" "${SLURM_JOB_ID:-unknown}" "$(hostname)" \
  >"$COMPLETED_MARKER"
