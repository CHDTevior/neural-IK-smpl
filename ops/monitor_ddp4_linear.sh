#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/iridisfs/scratch/ts1v23/workspace/learned_IK/smpl-ik"
OUT="$REPO_DIR/run_reports/ddp4_linear_monitor.tsv"
INTERVAL_SECONDS="${DDP_MONITOR_INTERVAL_SECONDS:-300}"

printf 'timestamp\tjob_id\thost\tphase\tgpu_index\tutil_gpu_pct\tmemory_used_mib\tmemory_total_mib\tcheckpoint\tcheckpoint_bytes\n' >"$OUT"

while true; do
  timestamp="$(date --iso-8601=seconds)"
  job_id="${SLURM_JOB_ID:-unknown}"
  host="$(hostname)"
  phase="waiting"
  if pgrep -af 'run.py --config=ops/smplik_amass_ddp4_linear_smoke.yaml' >/dev/null; then
    phase="smoke"
  elif pgrep -af 'run.py --config=ops/smplik_amass_ddp4_linear.yaml' >/dev/null; then
    phase="formal"
  fi

  checkpoint=""
  checkpoint_bytes="0"
  latest="$(
    find "$REPO_DIR/logs/protores_amass_ddp4_linear" -type f -name '*.ckpt' \
      -printf '%T@ %s %p\n' 2>/dev/null |
      sort -nr |
      head -n 1 || true
  )"
  if [[ -n "$latest" ]]; then
    checkpoint_bytes="$(awk '{print $2}' <<<"$latest")"
    checkpoint="$(cut -d' ' -f3- <<<"$latest")"
  fi

  while IFS=',' read -r index utilization memory_used memory_total; do
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$timestamp" "$job_id" "$host" "$phase" \
      "${index// /}" "${utilization// /}" "${memory_used// /}" "${memory_total// /}" \
      "$checkpoint" "$checkpoint_bytes" >>"$OUT"
  done < <(
    nvidia-smi \
      --query-gpu=index,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits
  )

  sleep "$INTERVAL_SECONDS"
done
