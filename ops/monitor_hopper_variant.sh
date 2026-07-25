#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
  echo "usage: monitor_hopper_variant.sh {h100x7|h200x4} ATTEMPT_ID [interval_seconds]" >&2
  exit 2
fi

VARIANT="$1"
ATTEMPT_ID="$2"
INTERVAL="${3:-60}"
if [[ ! "$ATTEMPT_ID" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Invalid attempt ID: $ATTEMPT_ID" >&2
  exit 2
fi

REPO_DIR="/iridisfs/scratch/ts1v23/workspace/learned_IK/smpl-ik"
REPORT_DIR="$REPO_DIR/run_reports"
ATTEMPT_DIR="$REPORT_DIR/${VARIANT}_formal_attempts/$ATTEMPT_ID"
ATTEMPT_STATUS="$ATTEMPT_DIR/status.json"
TRAIN_STEP_NAME="smplik-${VARIANT}-formal-${ATTEMPT_ID}"
POST_STEP_NAME="${TRAIN_STEP_NAME}-post"

case "$VARIANT" in
  h100x7)
    LOG_DIR="$REPO_DIR/logs/protores_amass_h100x7_linear/seed-0_h100x7_global_batch-14336_lr-0.0014"
    JOB_IDS=(993162 993160 993161 977978)
    NODES=(swarmh1001)
    ;;
  h200x4)
    LOG_DIR="$REPO_DIR/logs/protores_amass_h200x4_linear/seed-0_h200x4_global_batch-8192_lr-0.0008"
    JOB_IDS=(1266301 1266302)
    NODES=(blossom04 flamingo01)
    ;;
  *)
    echo "unknown variant: $VARIANT" >&2
    exit 2
    ;;
esac

PRIMARY_JOB="${JOB_IDS[0]}"
EXPECTED_TRAIN_STEPS="${#JOB_IDS[@]}"
MONITOR_LOG="$ATTEMPT_DIR/monitor.log"
STOP_MARKER="$LOG_DIR/postprocess_attempts/$ATTEMPT_ID.json"
if [[ ! -d "$ATTEMPT_DIR" ]]; then
  echo "Attempt directory does not exist: $ATTEMPT_DIR" >&2
  exit 1
fi

json_field() {
  local path="$1"
  local field="$2"
  sed -n "s/.*\"${field}\": \"\\([^\"]*\\)\".*/\\1/p" "$path" |
    head -1
}

train_step_count() {
  local count=0
  local job_id
  for job_id in "${JOB_IDS[@]}"; do
    count=$(
      (
        printf '%s\n' "$count"
        squeue --steps -h -j "$job_id" -o '%i|%.200j' 2>/dev/null |
          awk -F'|' -v train="$TRAIN_STEP_NAME" '
            {
              gsub(/^[ \t]+|[ \t]+$/, "", $2)
              if ($2 == train) print 1
            }
          '
      ) | awk '{sum += $1} END {print sum + 0}'
    )
  done
  printf '%s\n' "$count"
}

post_step_count() {
  squeue --steps -h -j "$PRIMARY_JOB" -o '%i|%.200j' 2>/dev/null |
    awk -F'|' -v post="$POST_STEP_NAME" '
      {
        gsub(/^[ \t]+|[ \t]+$/, "", $2)
        if ($2 == post) count += 1
      }
      END {print count + 0}
    '
}

inactive_polls=0
missing_step_polls=0
while true; do
  inactive_required=0
  {
    echo "time=$(date --iso-8601=seconds) variant=$VARIANT attempt_id=$ATTEMPT_ID"
    if [[ -s "$ATTEMPT_STATUS" ]]; then
      tr '\n' ' ' <"$ATTEMPT_STATUS"
      echo
    else
      echo "attempt_status=missing"
    fi
    if [[ -s "$LOG_DIR/run_state.json" ]]; then
      tr '\n' ' ' <"$LOG_DIR/run_state.json"
      echo
    else
      echo "run_state=missing"
    fi
    checkpoint="$(
      find "$LOG_DIR/checkpoints" -maxdepth 1 -type f -name '*.ckpt' -size +0c \
        -printf '%T@ %s %p\n' 2>/dev/null | sort -nr | head -1 || true
    )"
    echo "latest_checkpoint=${checkpoint:-none}"
    for job_id in "${JOB_IDS[@]}"; do
      job_state="$(squeue -h -j "$job_id" -o '%T' | head -1)"
      echo "job=$job_id state=${job_state:-absent}"
      if [[ "$job_state" != "RUNNING" ]]; then
        inactive_required=$((inactive_required + 1))
      fi
    done
    for node in "${NODES[@]}"; do
      if [[ "$(hostname -s)" == "$node" ]]; then
        gpu_output="$(
          nvidia-smi --query-gpu=uuid,name,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader 2>&1 || true
        )"
      else
        gpu_output="$(
          ssh -o BatchMode=yes -o ConnectTimeout=5 "$node" \
            nvidia-smi --query-gpu=uuid,name,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader 2>&1 || true
        )"
      fi
      while IFS= read -r gpu_line; do
      echo "gpu_node=$node value=$gpu_line"
      done <<<"$gpu_output"
    done
    train_steps="$(train_step_count)"
    post_steps="$(post_step_count)"
    echo "train_steps=$train_steps expected_train_steps=$EXPECTED_TRAIN_STEPS post_steps=$post_steps"
  } >>"$MONITOR_LOG"

  if [[ -s "$ATTEMPT_STATUS" ]]; then
    status_attempt_id="$(json_field "$ATTEMPT_STATUS" attempt_id)"
    if [[ "$status_attempt_id" != "$ATTEMPT_ID" ]]; then
      echo "time=$(date --iso-8601=seconds) stop=status_attempt_mismatch" >>"$MONITOR_LOG"
      exit 1
    fi
  fi

  if [[ -s "$STOP_MARKER" ]]; then
    marker_attempt_id="$(json_field "$STOP_MARKER" attempt_id)"
    if [[ "$marker_attempt_id" != "$ATTEMPT_ID" ]]; then
      echo "time=$(date --iso-8601=seconds) stop=marker_attempt_mismatch" >>"$MONITOR_LOG"
      exit 1
    fi
  fi

  if [[ -s "$ATTEMPT_STATUS" ]]; then
    launcher_status="$(json_field "$ATTEMPT_STATUS" status)"
    if [[ "$launcher_status" == "FAILED" ]]; then
      echo "time=$(date --iso-8601=seconds) stop=launcher_failed" >>"$MONITOR_LOG"
      exit 1
    fi
    if [[ "$launcher_status" == "COMPLETED" ]]; then
      if [[ -s "$STOP_MARKER" ]]; then
        echo "time=$(date --iso-8601=seconds) stop=postprocess_completed" >>"$MONITOR_LOG"
        exit 0
      else
        echo "time=$(date --iso-8601=seconds) stop=completed_without_marker" >>"$MONITOR_LOG"
        exit 1
      fi
    fi
  fi

  if [[ "$inactive_required" -gt 0 ]]; then
    inactive_polls=$((inactive_polls + 1))
  else
    inactive_polls=0
  fi
  if [[ "$inactive_polls" -ge 3 ]]; then
    echo "time=$(date --iso-8601=seconds) stop=required_allocation_inactive" >>"$MONITOR_LOG"
    exit 1
  fi

  if [[ "$train_steps" -eq "$EXPECTED_TRAIN_STEPS" || "$post_steps" -eq 1 ]]; then
    missing_step_polls=0
  else
    missing_step_polls=$((missing_step_polls + 1))
  fi
  if [[ "$missing_step_polls" -ge 3 ]]; then
    echo "time=$(date --iso-8601=seconds) stop=attempt_steps_absent" >>"$MONITOR_LOG"
    exit 1
  fi

  sleep "$INTERVAL"
done
