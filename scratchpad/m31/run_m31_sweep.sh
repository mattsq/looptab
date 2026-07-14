#!/bin/zsh
# M31 shared-readout-confound sweep — RESUMABLE runner (mirrors M30's).
#
# Runs the six lean configs (etth1/weather x {192,336,720}), each a full 10-seed budget-matched
# run (4 parallel workers, 4 arms). SAFE TO RE-RUN: a config is SKIPPED if a completed
# `results/m31_<cfg>_*_deltas.csv` already exists, so this resumes an interrupted sweep.
#
# Usage:  ./scratchpad/m31/run_m31_sweep.sh          # resume (skip completed)
#         FORCE=1 ./scratchpad/m31/run_m31_sweep.sh  # re-run everything
#
# SLEEP GUARD: re-execs under `caffeinate -ims` so macOS idle/lid sleep can't suspend the workers
# (M30 lost ~18h that way). `-s` (system sleep) only holds on AC — keep the laptop plugged in.

if [[ -z "$_M31_CAFFEINATED" ]]; then
  export _M31_CAFFEINATED=1
  exec caffeinate -ims "$0" "$@"
fi

cd "$(dirname "$0")/../.." || exit 1
LOG=scratchpad/m31/m31_sweep.log
# etth1 (cheap, 7 vars) first so the decomposition lands early; weather (21 vars, heavy) last.
CONFIGS=(m31_etth1_h192 m31_etth1_h336 m31_etth1_h720 m31_weather_h192 m31_weather_h336 m31_weather_h720)

echo "===== SWEEP START $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$LOG"
for cfg in $CONFIGS; do
  if [[ -z "$FORCE" ]] && ls results/${cfg}_*_deltas.csv >/dev/null 2>&1; then
    echo "===== SKIP  $cfg (already complete) $(date '+%H:%M:%S') =====" >> "$LOG"
    continue
  fi
  echo "===== START $cfg $(date '+%H:%M:%S') =====" >> "$LOG"
  uv run python -m looptab.run --config configs/experiments/$cfg.yaml >> "$LOG" 2>&1
  echo "===== DONE  $cfg rc=$? $(date '+%H:%M:%S') =====" >> "$LOG"
done
echo "===== ALL_DONE $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$LOG"
