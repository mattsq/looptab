#!/bin/zsh
# M30 forecasting horizon sweep — RESUMABLE runner.
#
# Runs the six per-horizon configs (etth1/weather x {192,336,720}). Each config is a full
# 10-seed, budget-matched run (4 parallel workers). SAFE TO RE-RUN: a config is SKIPPED if a
# completed `results/m30_<cfg>_*_deltas.csv` already exists, so this resumes an interrupted sweep.
#
# Progress so far (run 1, 2026-07-10): DONE = etth1_h192, etth1_h336, etth1_h720, weather_h192
# (all budget-clean). REMAINING = weather_h336, weather_h720 (weather is ~2-5h EACH — heavy:
# 21 vars, mixer hidden 1200/1592, big untied ceiling). Run 1 log: scratchpad/m30/m30_sweep_run1.log
#
# Usage:  ./scratchpad/m30/run_m30_sweep.sh            # resume (skip completed)
#         FORCE=1 ./scratchpad/m30/run_m30_sweep.sh    # re-run everything from scratch
#
# Runs in the FOREGROUND of whatever shell launches it; launch detached if you want to walk away.
#
# SLEEP GUARD: macOS idle/lid sleep will SUSPEND the workers overnight (run-1 lost ~18h this way —
# 25h wall-clock bought ~25min CPU). This script re-execs itself under `caffeinate -ims` so the
# machine stays awake for the WHOLE sweep and the assertion auto-releases when it exits. `-s`
# (system sleep) only holds on AC power, so keep the laptop plugged in for long weather configs.

if [[ -z "$_M30_CAFFEINATED" ]]; then
  export _M30_CAFFEINATED=1
  exec caffeinate -ims "$0" "$@"           # re-exec self held awake; releases on exit
fi

cd "$(dirname "$0")/../.." || exit 1        # repo root
LOG=scratchpad/m30/m30_sweep.log
CONFIGS=(m30_etth1_h192 m30_etth1_h336 m30_etth1_h720 m30_weather_h192 m30_weather_h336 m30_weather_h720)

echo "===== SWEEP START $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$LOG"
for cfg in $CONFIGS; do
  # resume guard: skip if a completed deltas file already exists (unless FORCE=1)
  if [[ -z "$FORCE" ]] && ls results/${cfg}_*_deltas.csv >/dev/null 2>&1; then
    echo "===== SKIP  $cfg (already complete) $(date '+%H:%M:%S') =====" >> "$LOG"
    continue
  fi
  echo "===== START $cfg $(date '+%H:%M:%S') =====" >> "$LOG"
  uv run python -m looptab.run --config configs/experiments/$cfg.yaml >> "$LOG" 2>&1
  echo "===== DONE  $cfg rc=$? $(date '+%H:%M:%S') =====" >> "$LOG"
done
echo "===== ALL_DONE $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$LOG"
