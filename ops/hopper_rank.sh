#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 8 ]]; then
  echo "usage: hopper_rank.sh RANK_BASE WORLD_SIZE MASTER_ADDR MASTER_PORT CONFIG VARIANT PHASE LOG_ROOT" >&2
  exit 2
fi

RANK_BASE="$1"
WORLD_SIZE_VALUE="$2"
MASTER_ADDR_VALUE="$3"
MASTER_PORT_VALUE="$4"
CONFIG_PATH="$5"
VARIANT="$6"
PHASE="$7"
LOG_ROOT="$8"

REPO_DIR="/iridisfs/scratch/ts1v23/workspace/learned_IK/smpl-ik"
ENV_DIR="/iridisfs/scratch/ts1v23/workspace/learned_IK/.envs/smplik-hopper-py38-torch24-cu124"

if [[ -z "${SLURM_PROCID:-}" ]]; then
  echo "SLURM_PROCID is required" >&2
  exit 2
fi

GLOBAL_RANK=$((RANK_BASE + SLURM_PROCID))
RANK_LOG="$LOG_ROOT/rank-${GLOBAL_RANK}.log"
exec > >(tee "$RANK_LOG") 2>&1

export PATH="$ENV_DIR/bin:$PATH"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Each Slurm task is cgroup-scoped to one physical GPU. Treat it as a
# one-GPU logical node so Lightning 1.4 selects TorchElasticEnvironment.
export RANK="$GLOBAL_RANK"
export GLOBAL_RANK="$GLOBAL_RANK"
export WORLD_SIZE="$WORLD_SIZE_VALUE"
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1
export GROUP_RANK="$GLOBAL_RANK"
export NODE_RANK="$GLOBAL_RANK"
export MASTER_ADDR="$MASTER_ADDR_VALUE"
export MASTER_PORT="$MASTER_PORT_VALUE"
export HOPPER_PHASE="$PHASE"
export HOPPER_EVIDENCE_ROOT="$LOG_ROOT"

# Separate Slurm allocations and per-task GPU cgroups on the same physical
# host cannot inspect one another through NVML. Give NCCL a rank-unique host
# identity so it uses the verified IB transport instead of invalid local-GPU
# topology discovery across those cgroup boundaries.
export NCCL_HOSTID="smplik-${VARIANT}-rank-${RANK}"

# Lightning 1.4 installs a Slurm SIGTERM handler that intentionally ignores
# termination and a SIGUSR1 handler that requeues the enclosing allocation.
# These ranks are externally orchestrated TorchElastic processes spanning
# several allocations, so disable only that legacy Slurm integration.
export SMPLIK_ORIGINAL_SLURM_JOB_NAME="${SLURM_JOB_NAME:-unset}"
export SLURM_JOB_NAME=bash

export NCCL_SOCKET_IFNAME=ib
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
if [[ "$PHASE" == "smoke" ]]; then
  export TORCH_DISTRIBUTED_DEBUG=DETAIL
  export HOPPER_LOCAL_GRAD_PROBE=1
else
  export TORCH_DISTRIBUTED_DEBUG=OFF
  unset HOPPER_LOCAL_GRAD_PROBE
fi

cd "$REPO_DIR"

echo "time=$(date --iso-8601=seconds)"
echo "variant=$VARIANT phase=$PHASE rank=$RANK world_size=$WORLD_SIZE"
echo "host=$(hostname) slurm_job=${SLURM_JOB_ID:-unknown} slurm_procid=$SLURM_PROCID"
echo "attempt_id=${HOPPER_ATTEMPT_ID:-manual} original_slurm_job_name=$SMPLIK_ORIGINAL_SLURM_JOB_NAME"
echo "master=$MASTER_ADDR:$MASTER_PORT"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "command=python ops/run_hopper_ddp.py --config=$CONFIG_PATH"
nvidia-smi --query-gpu=uuid,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader

exec python ops/run_hopper_ddp.py --config="$CONFIG_PATH"
