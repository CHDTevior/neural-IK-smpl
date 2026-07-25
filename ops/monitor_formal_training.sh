#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/iridisfs/scratch/ts1v23/workspace/learned_IK/smpl-ik"
FORMAL_DIR="$REPO_DIR/logs/protores_amass/seed-0_num_blocks_stage1-3"
FORMAL_LOG="$REPO_DIR/run_reports/formal_training.log"
MONITOR_LOG="$REPO_DIR/run_reports/formal_monitor.tsv"
LATEST_STATUS="$REPO_DIR/run_reports/formal_monitor.latest"
COMPLETED_MARKER="$REPO_DIR/run_reports/formal_training.completed"
INTERVAL_SECONDS="${SMPLIK_MONITOR_INTERVAL_SECONDS:-900}"
MONITORED_JOB_ID="${SMPLIK_MONITORED_SLURM_JOB_ID:-${SLURM_JOB_ID:-}}"
FORMAL_COMMAND='python run.py --config=configs/experiments/smplik_amass.yaml'

if [[ ! "$INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "SMPLIK_MONITOR_INTERVAL_SECONDS must be a positive integer" >&2
  exit 1
fi
if [[ -z "$MONITORED_JOB_ID" ]]; then
  echo "Set SMPLIK_MONITORED_SLURM_JOB_ID to the formal Slurm job" >&2
  exit 1
fi

cd "$REPO_DIR"
exec 9>"$REPO_DIR/run_reports/formal_monitor.lock"
if ! flock -n 9; then
  echo "Another formal-training monitor already holds the lock" >&2
  exit 1
fi

if [[ ! -e "$MONITOR_LOG" ]]; then
  printf 'timestamp\tstate\tpid\tprogress\tgpu\tcheckpoint\n' >"$MONITOR_LOG"
fi

expected_uid="$(id -u)"
candidate_pids=()
while IFS= read -r candidate_pid; do
  [[ -n "$candidate_pid" ]] || continue
  candidate_uid="$(
    awk '/^Uid:/ {print $2; exit}' "/proc/$candidate_pid/status" 2>/dev/null ||
      true
  )"
  candidate_cwd="$(readlink -f "/proc/$candidate_pid/cwd" 2>/dev/null || true)"
  candidate_job="$(
    tr '\0' '\n' <"/proc/$candidate_pid/environ" 2>/dev/null |
      sed -n 's/^SLURM_JOB_ID=//p' |
      head -n 1 || true
  )"
  if [[ "$candidate_uid" == "$expected_uid" ]] &&
     [[ "$candidate_cwd" == "$REPO_DIR" ]] &&
     [[ "$candidate_job" == "$MONITORED_JOB_ID" ]]; then
    candidate_pids+=("$candidate_pid")
  fi
done < <(pgrep -f "^${FORMAL_COMMAND}$" || true)

controller_pids=()
for candidate_pid in "${candidate_pids[@]}"; do
  candidate_ppid="$(
    awk '/^PPid:/ {print $2; exit}' "/proc/$candidate_pid/status" 2>/dev/null ||
      true
  )"
  parent_is_candidate=false
  for possible_parent in "${candidate_pids[@]}"; do
    if [[ "$candidate_ppid" == "$possible_parent" ]]; then
      parent_is_candidate=true
      break
    fi
  done
  if [[ "$parent_is_candidate" == false ]]; then
    controller_pids+=("$candidate_pid")
  fi
done
if [[ "${#controller_pids[@]}" -ne 1 ]]; then
  echo "Expected one scoped formal-training controller, found ${#controller_pids[@]}" >&2
  exit 1
fi
training_pid="${controller_pids[0]}"

while true; do
  timestamp="$(date --iso-8601=seconds)"
  progress="$(
    tail -c 1048576 "$FORMAL_LOG" 2>/dev/null |
      tr '\r' '\n' |
      awk '/Epoch [0-9]+:/ {line=$0} END {print line}' |
      tr '\t' ' ' || true
  )"
  gpu="$(
    nvidia-smi \
      --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw \
      --format=csv,noheader 2>/dev/null |
      head -n 1 || true
  )"
  checkpoint="$(
    find "$FORMAL_DIR/checkpoints" -maxdepth 1 -type f -name '*.ckpt' \
      -printf '%T@ %s %p\n' 2>/dev/null |
      sort -nr |
      head -n 1 || true
  )"

  if kill -0 "$training_pid" 2>/dev/null; then
    state="RUNNING"
  else
    sleep 2
    if [[ -s "$COMPLETED_MARKER" ]]; then
      state="COMPLETED"
    else
      state="STOPPED"
    fi
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$timestamp" "$state" "${training_pid:-none}" "$progress" "$gpu" \
    "${checkpoint:-none}" >>"$MONITOR_LOG"
  printf 'timestamp=%s\nstate=%s\npid=%s\nprogress=%s\ngpu=%s\ncheckpoint=%s\n' \
    "$timestamp" "$state" "${training_pid:-none}" "$progress" "$gpu" \
    "${checkpoint:-none}" >"$LATEST_STATUS"

  if [[ "$state" != "RUNNING" ]]; then
    exit 0
  fi
  sleep "$INTERVAL_SECONDS"
done
