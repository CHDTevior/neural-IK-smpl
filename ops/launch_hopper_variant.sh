#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: launch_hopper_variant.sh {h100x7|h200x4} {smoke|formal}" >&2
  exit 2
fi

VARIANT="$1"
PHASE="$2"
if [[ "$PHASE" != "smoke" && "$PHASE" != "formal" ]]; then
  echo "phase must be smoke or formal" >&2
  exit 2
fi

REPO_DIR="/iridisfs/scratch/ts1v23/workspace/learned_IK/smpl-ik"
ENV_DIR="/iridisfs/scratch/ts1v23/workspace/learned_IK/.envs/smplik-hopper-py38-torch24-cu124"
REPORT_DIR="$REPO_DIR/run_reports"
STATUS_PATH="$REPORT_DIR/${VARIANT}_${PHASE}_launcher_status.json"
ATTEMPT_ID="${HOPPER_ATTEMPT_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
if [[ ! "$ATTEMPT_ID" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Invalid HOPPER_ATTEMPT_ID: $ATTEMPT_ID" >&2
  exit 2
fi
export HOPPER_ATTEMPT_ID="$ATTEMPT_ID"

case "$VARIANT" in
  h100x7)
    WORLD_SIZE_VALUE=7
    MASTER_ADDR_VALUE="swarmh1001-ib0"
    if [[ "$PHASE" == "smoke" ]]; then
      MASTER_PORT_VALUE=29671
      CONFIG_PATH="ops/smplik_amass_h100x7_linear_smoke.yaml"
      LOG_DIR="$REPO_DIR/logs/protores_amass_h100x7_linear_smoke/seed-0_h100x7_global_batch-14336_lr-0.0014"
    else
      MASTER_PORT_VALUE=29672
      CONFIG_PATH="ops/smplik_amass_h100x7_linear.yaml"
      LOG_DIR="$REPO_DIR/logs/protores_amass_h100x7_linear/seed-0_h100x7_global_batch-14336_lr-0.0014"
    fi
    JOB_IDS=(993162 993160 993161 977978)
    TASK_COUNTS=(2 2 2 1)
    CPU_COUNTS=(16 16 16 8)
    RANK_BASES=(0 2 4 6)
    ;;
  h200x4)
    WORLD_SIZE_VALUE=4
    MASTER_ADDR_VALUE="blossom04-ib0"
    if [[ "$PHASE" == "smoke" ]]; then
      MASTER_PORT_VALUE=29641
      CONFIG_PATH="ops/smplik_amass_h200x4_linear_smoke.yaml"
      LOG_DIR="$REPO_DIR/logs/protores_amass_h200x4_linear_smoke/seed-0_h200x4_global_batch-8192_lr-0.0008"
    else
      MASTER_PORT_VALUE=29642
      CONFIG_PATH="ops/smplik_amass_h200x4_linear.yaml"
      LOG_DIR="$REPO_DIR/logs/protores_amass_h200x4_linear/seed-0_h200x4_global_batch-8192_lr-0.0008"
    fi
    JOB_IDS=(1266301 1266302)
    TASK_COUNTS=(2 2)
    CPU_COUNTS=(4 4)
    RANK_BASES=(0 2)
    ;;
  *)
    echo "unknown variant: $VARIANT" >&2
    exit 2
    ;;
esac

ATTEMPT_PARENT="$REPORT_DIR/${VARIANT}_${PHASE}_attempts"
ATTEMPT_DIR="$ATTEMPT_PARENT/$ATTEMPT_ID"
ATTEMPT_STATUS_PATH="$ATTEMPT_DIR/status.json"
RANK_LOG_ROOT="$ATTEMPT_DIR/ranks"
POSTPROCESS_LOG="$ATTEMPT_DIR/postprocess.log"
STOP_MARKER="$LOG_DIR/postprocess_attempts/$ATTEMPT_ID.json"
POSTPROCESS_RESULT="$LOG_DIR/postprocess_results/$ATTEMPT_ID.json"
TRAIN_STEP_NAME="smplik-${VARIANT}-${PHASE}-${ATTEMPT_ID}"
POST_STEP_NAME="${TRAIN_STEP_NAME}-post"

cd "$REPO_DIR"
mkdir -p "$REPORT_DIR" "$ATTEMPT_PARENT"
if [[ -e "$STOP_MARKER" || -e "$POSTPROCESS_RESULT" ]]; then
  echo "Attempt artifacts already exist for $ATTEMPT_ID" >&2
  exit 1
fi
if ! mkdir "$ATTEMPT_DIR"; then
  echo "Attempt directory already exists: $ATTEMPT_DIR" >&2
  exit 1
fi
mkdir -p "$RANK_LOG_ROOT"

# Prevent two launchers from driving the same experiment/log directory.
exec 9>"$REPORT_DIR/${VARIANT}_${PHASE}.lock"
if ! flock -n 9; then
  echo "Another $VARIANT $PHASE launcher holds the lock" >&2
  exit 1
fi

write_one_status() {
  local destination="$1"
  local status="$2"
  local exit_code="$3"
  local signal_name="$4"
  local temporary="${destination}.tmp.$$"
  printf '{\n  "time": "%s",\n  "attempt_id": "%s",\n  "variant": "%s",\n  "phase": "%s",\n  "status": "%s",\n  "exit_code": %s,\n  "signal": "%s",\n  "stop_marker": "%s"\n}\n' \
    "$(date --iso-8601=seconds)" "$ATTEMPT_ID" "$VARIANT" "$PHASE" \
    "$status" "$exit_code" "$signal_name" "$STOP_MARKER" >"$temporary"
  mv -f "$temporary" "$destination"
}

write_status() {
  write_one_status "$ATTEMPT_STATUS_PATH" "$1" "$2" "$3"
  write_one_status "$STATUS_PATH" "$1" "$2" "$3"
}

pids=()
cleanup_started=0
EXIT_SIGNAL=""

matching_step_ids() {
  local job_id="$1"
  squeue --steps -h -j "$job_id" -o '%i|%.200j' 2>/dev/null |
    awk -F'|' -v train="$TRAIN_STEP_NAME" -v post="$POST_STEP_NAME" '
      {
        gsub(/^[ \t]+|[ \t]+$/, "", $1)
        gsub(/^[ \t]+|[ \t]+$/, "", $2)
        if ($2 == train || $2 == post) print $1
      }
    '
}

cancel_named_steps() {
  local signal_name="$1"
  local job_id step_id
  for job_id in "${JOB_IDS[@]}"; do
    while IFS= read -r step_id; do
      [[ -n "$step_id" ]] || continue
      scancel --signal="$signal_name" "$step_id" 2>/dev/null || true
    done < <(matching_step_ids "$job_id")
  done
}

cleanup_children() {
  [[ "$cleanup_started" -eq 0 ]] || return
  cleanup_started=1
  set +e

  if [[ "${#pids[@]}" -gt 0 ]]; then
    kill -TERM "${pids[@]}" 2>/dev/null || true
  fi
  cancel_named_steps TERM

  local deadline=$((SECONDS + 10))
  local alive pid
  while [[ "$SECONDS" -lt "$deadline" ]]; do
    alive=0
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        alive=1
      fi
    done
    [[ "$alive" -eq 1 ]] || break
    sleep 0.2
  done

  if [[ "${#pids[@]}" -gt 0 ]]; then
    kill -KILL "${pids[@]}" 2>/dev/null || true
  fi
  cancel_named_steps KILL
  wait "${pids[@]}" 2>/dev/null || true
}

on_signal() {
  EXIT_SIGNAL="$1"
  case "$1" in
    HUP) exit 129 ;;
    INT) exit 130 ;;
    TERM) exit 143 ;;
    *) exit 1 ;;
  esac
}

on_exit() {
  local exit_code=$?
  trap - EXIT HUP INT TERM
  set +e

  if [[ "$exit_code" -eq 0 && ! -s "$STOP_MARKER" ]]; then
    echo "Successful launcher exit without attempt postprocess marker: $STOP_MARKER" >&2
    exit_code=1
  fi
  if [[ "$exit_code" -ne 0 ]]; then
    cleanup_children
    write_status "FAILED" "$exit_code" "$EXIT_SIGNAL"
  else
    write_status "COMPLETED" 0 ""
  fi
  exit "$exit_code"
}

trap 'on_signal HUP' HUP
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
trap on_exit EXIT
write_status "STARTING" 0 ""

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "Hopper environment is unavailable: $ENV_DIR" >&2
  exit 1
fi

if ! printf '%s\n' \
  '07621dd6d361f6db4e813ddb974f9a9364e3c1468c66cbad1c3d89ebdc537d06  configs/experiments/smplik_amass.yaml' \
  '7b7a9bc62b44e697a60e61d43e07397595d4db4be3162dfad792d857ee06b549  configs/posing_smpl.yaml' \
  'b8ac2f94b971aeca344cbdc99d42221a75e2d58682bc119f275195decb17d9eb  configs/model/posing_smpl_default.yaml' \
  'a3c894a62fb441d37f423a777566062d6dcfebc06dbe5f6da8440d8e8d0156dc  smplik/models/smpl_model.py' |
  sha256sum --check --status; then
  echo "Protected source/config checksum mismatch" >&2
  exit 1
fi
if ! git diff --quiet -- \
  configs/experiments/smplik_amass.yaml \
  configs/posing_smpl.yaml \
  configs/model/posing_smpl_default.yaml \
  smplik/models/smpl_model.py; then
  echo "Protected source/config has a Git diff" >&2
  exit 1
fi

"$ENV_DIR/bin/python" ops/audit_hopper_configs.py \
  --output run_reports/hopper_config_audit.json >/dev/null

total_tasks=0
for index in "${!JOB_IDS[@]}"; do
  job_id="${JOB_IDS[$index]}"
  if [[ "$(squeue -h -j "$job_id" -t R -o '%i')" != "$job_id" ]]; then
    echo "Required allocation is not running: $job_id" >&2
    exit 1
  fi
  total_tasks=$((total_tasks + TASK_COUNTS[index]))
done
if [[ "$total_tasks" -ne "$WORLD_SIZE_VALUE" ]]; then
  echo "Task map totals $total_tasks, expected $WORLD_SIZE_VALUE" >&2
  exit 1
fi

echo "time=$(date --iso-8601=seconds)"
echo "attempt_id=$ATTEMPT_ID"
echo "classification=linear-scaled approximation, not canonical exact reproduction"
echo "variant=$VARIANT phase=$PHASE world_size=$WORLD_SIZE_VALUE"
echo "config=$CONFIG_PATH"
echo "master=$MASTER_ADDR_VALUE:$MASTER_PORT_VALUE"
echo "jobs=${JOB_IDS[*]}"
echo "formal_python_command=python ops/run_hopper_ddp.py --config=$CONFIG_PATH"

for index in "${!JOB_IDS[@]}"; do
  job_id="${JOB_IDS[$index]}"
  task_count="${TASK_COUNTS[$index]}"
  cpu_count="${CPU_COUNTS[$index]}"
  rank_base="${RANK_BASES[$index]}"
  srun \
    --jobid="$job_id" \
    --job-name="$TRAIN_STEP_NAME" \
    --overlap \
    --kill-on-bad-exit=1 \
    --nodes=1 \
    --ntasks="$task_count" \
    --gpus-per-task=1 \
    --cpus-per-task="$cpu_count" \
    --cpu-bind=cores \
    --unbuffered \
    bash "$REPO_DIR/ops/hopper_rank.sh" \
      "$rank_base" \
      "$WORLD_SIZE_VALUE" \
      "$MASTER_ADDR_VALUE" \
      "$MASTER_PORT_VALUE" \
      "$CONFIG_PATH" \
      "$VARIANT" \
      "$PHASE" \
      "$RANK_LOG_ROOT" &
  pids+=("$!")
done
write_status "RUNNING" 0 ""

remaining="${#pids[@]}"
while [[ "$remaining" -gt 0 ]]; do
  if wait -n; then
    :
  else
    exit_code=$?
    echo "A rank group failed with exit code $exit_code" >&2
    exit "$exit_code"
  fi
  remaining=$((remaining - 1))
done

PRIMARY_JOB="${JOB_IDS[0]}"
POSTPROCESS_CPUS=8
POST_PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
echo "postprocess_command=python ops/postprocess_hopper.py --config=$CONFIG_PATH"
srun \
  --jobid="$PRIMARY_JOB" \
  --job-name="$POST_STEP_NAME" \
  --overlap \
  --kill-on-bad-exit=1 \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-task=1 \
  --cpus-per-task="$POSTPROCESS_CPUS" \
  --unbuffered \
  env \
    PATH="$ENV_DIR/bin:$PATH" \
    PYTHONPATH="$POST_PYTHONPATH" \
    PYTHONUNBUFFERED=1 \
    HOPPER_ATTEMPT_ID="$ATTEMPT_ID" \
    SLURM_JOB_NAME=bash \
    GLOBAL_RANK=0 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    "$ENV_DIR/bin/python" \
    "$REPO_DIR/ops/postprocess_hopper.py" \
    --config="$CONFIG_PATH" \
  > >(tee "$POSTPROCESS_LOG") 2>&1 &
post_pid=$!
pids+=("$post_pid")
if wait "$post_pid"; then
  :
else
  exit_code=$?
  echo "Postprocessing failed with exit code $exit_code" >&2
  exit "$exit_code"
fi
if [[ ! -s "$POSTPROCESS_RESULT" ]]; then
  echo "Postprocessing exited zero without result: $POSTPROCESS_RESULT" >&2
  exit 1
fi
result_attempt_id="$(
  "$ENV_DIR/bin/python" -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["attempt_id"])' \
    "$POSTPROCESS_RESULT"
)"
if [[ "$result_attempt_id" != "$ATTEMPT_ID" ]]; then
  echo "Postprocess result attempt mismatch: $result_attempt_id" >&2
  exit 1
fi
mkdir -p "$(dirname "$STOP_MARKER")"
marker_temporary="${STOP_MARKER}.tmp.$$"
cp "$POSTPROCESS_RESULT" "$marker_temporary"
mv -f "$marker_temporary" "$STOP_MARKER"
