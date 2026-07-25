#!/usr/bin/env bash
#SBATCH --job-name=smplik_ddp4_linear
#SBATCH --partition=swarm_a100
#SBATCH --account=ecs
#SBATCH --qos=ecsgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=450G
#SBATCH --time=5-00:00:00
#SBATCH --output=/iridisfs/scratch/ts1v23/workspace/learned_IK/smpl-ik/run_reports/ddp4_linear_slurm_%j.log
#SBATCH --error=/iridisfs/scratch/ts1v23/workspace/learned_IK/smpl-ik/run_reports/ddp4_linear_slurm_%j.log

set -euo pipefail

REPO_DIR="/iridisfs/scratch/ts1v23/workspace/learned_IK/smpl-ik"
ENV_DIR="/iridisfs/scratch/ts1v23/workspace/learned_IK/.envs/smplik-py38"
REPORT_DIR="$REPO_DIR/run_reports"
SMOKE_DIR="$REPO_DIR/logs/protores_amass_ddp4_linear_smoke/seed-0_ddp4_global_batch-8192_lr-0.0008"
FORMAL_DIR="$REPO_DIR/logs/protores_amass_ddp4_linear/seed-0_ddp4_global_batch-8192_lr-0.0008"
SMOKE_MARKER="$REPORT_DIR/ddp4_linear_smoke.success"
COMPLETED_MARKER="$REPORT_DIR/ddp4_linear_training.completed"
FAILED_MARKER="$REPORT_DIR/ddp4_linear_training.failed"

cd "$REPO_DIR"
mkdir -p "$REPORT_DIR"
rm -f "$FAILED_MARKER"

on_exit() {
  status=$?
  if [[ -n "${monitor_pid:-}" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  if [[ "$status" -ne 0 ]]; then
    printf 'time=%s job_id=%s host=%s exit_code=%s\n' \
      "$(date --iso-8601=seconds)" "${SLURM_JOB_ID:-unknown}" "$(hostname)" "$status" \
      >"$FAILED_MARKER"
  fi
}
trap on_exit EXIT

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "Pinned SMPL-IK Python is unavailable: $ENV_DIR/bin/python" >&2
  exit 1
fi
if ! printf '%s\n' \
  '07621dd6d361f6db4e813ddb974f9a9364e3c1468c66cbad1c3d89ebdc537d06  configs/experiments/smplik_amass.yaml' \
  '7b7a9bc62b44e697a60e61d43e07397595d4db4be3162dfad792d857ee06b549  configs/posing_smpl.yaml' \
  'b8ac2f94b971aeca344cbdc99d42221a75e2d58682bc119f275195decb17d9eb  configs/model/posing_smpl_default.yaml' \
  'a3c894a62fb441d37f423a777566062d6dcfebc06dbe5f6da8440d8e8d0156dc  smplik/models/smpl_model.py' |
  sha256sum --check --status; then
  echo "Protected training file checksum mismatch; DDP launch blocked" >&2
  exit 1
fi
if ! git diff --quiet -- \
  configs/experiments/smplik_amass.yaml \
  configs/posing_smpl.yaml \
  configs/model/posing_smpl_default.yaml \
  smplik/models/smpl_model.py; then
  echo "Protected training files changed; DDP launch blocked" >&2
  exit 1
fi
if [[ "$(nvidia-smi -L | wc -l)" -ne 4 ]]; then
  echo "Exactly four visible GPUs are required" >&2
  nvidia-smi -L >&2
  exit 1
fi

export PATH="$ENV_DIR/bin:$PATH"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

echo "DDP allocation start: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Slurm job: ${SLURM_JOB_ID:-unknown}"
echo "Classification: linear-scaled approximation, not canonical exact reproduction"
echo "Scaling: per-rank batch=2048, world-size=4, global batch=8192, lr=0.0008"
echo "Formal command: python run.py --config=ops/smplik_amass_ddp4_linear.yaml"
nvidia-smi
free -h

python ops/audit_ddp_linear_config.py \
  --canonical configs/experiments/smplik_amass.yaml \
  --formal ops/smplik_amass_ddp4_linear.yaml \
  --smoke ops/smplik_amass_ddp4_linear_smoke.yaml \
  --training-rows 50480676 \
  --output run_reports/ddp4_linear_config_audit.json

bash ops/monitor_ddp4_linear.sh &
monitor_pid=$!

if [[ ! -s "$SMOKE_MARKER" ]]; then
  if [[ -e "$SMOKE_DIR/hparams.yaml" ]]; then
    echo "Incomplete prior smoke output exists without success marker: $SMOKE_DIR" >&2
    exit 1
  fi
  echo "DDP smoke start: $(date --iso-8601=seconds)"
  echo "Smoke command: python run.py --config=ops/smplik_amass_ddp4_linear_smoke.yaml"
  python run.py --config=ops/smplik_amass_ddp4_linear_smoke.yaml \
    2>&1 | tee "$REPORT_DIR/ddp4_linear_smoke.log"

  smoke_checkpoint="$(
    find "$SMOKE_DIR/checkpoints" -maxdepth 1 -type f -name '*.ckpt' -size +0c \
      -print -quit 2>/dev/null || true
  )"
  if [[ -z "$smoke_checkpoint" || ! -s "$SMOKE_DIR/model.onnx" ]]; then
    echo "DDP smoke did not produce both a checkpoint and ONNX artifact" >&2
    exit 1
  fi
  printf 'time=%s job_id=%s host=%s checkpoint=%s onnx=%s\n' \
    "$(date --iso-8601=seconds)" "${SLURM_JOB_ID:-unknown}" "$(hostname)" \
    "$smoke_checkpoint" "$SMOKE_DIR/model.onnx" >"$SMOKE_MARKER"
fi

echo "DDP formal start: $(date --iso-8601=seconds)"
echo "Exact command: python run.py --config=ops/smplik_amass_ddp4_linear.yaml"
python run.py --config=ops/smplik_amass_ddp4_linear.yaml \
  2>&1 | tee -a "$REPORT_DIR/ddp4_linear_formal.log"

formal_checkpoint="$(
  find "$FORMAL_DIR/checkpoints" -maxdepth 1 -type f -name '*.ckpt' -size +0c \
    -print -quit 2>/dev/null || true
)"
if [[ -z "$formal_checkpoint" || ! -s "$FORMAL_DIR/model.onnx" ]]; then
  echo "DDP formal run returned without both a checkpoint and final ONNX artifact" >&2
  exit 1
fi

printf 'time=%s job_id=%s host=%s checkpoint=%s onnx=%s\n' \
  "$(date --iso-8601=seconds)" "${SLURM_JOB_ID:-unknown}" "$(hostname)" \
  "$formal_checkpoint" "$FORMAL_DIR/model.onnx" >"$COMPLETED_MARKER"
