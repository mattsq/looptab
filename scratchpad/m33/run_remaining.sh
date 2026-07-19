#!/usr/bin/env bash
# M33 RESUME script — run the M33 decomposition configs that have not completed yet.
#
# Idempotent: for each config it checks whether a results/<config>_*_deltas.csv already exists and
# SKIPS it if so; otherwise it runs it. So it is safe to re-run at any time — it only does outstanding
# work. (run.py writes its CSVs only on full completion, so an interrupted config leaves no deltas file
# and is correctly re-run from scratch.)
#
# State at time of writing (2026-07-16, after a manual shutdown mid-mixed_converge_w32):
#   DONE : converge_w24, converge_w32, converge_w48, hopfield_w24, hopfield_w32, mixed_converge_w24
#   TODO : mixed_converge_w32, disruption_w24, disruption_w32, sudoku
# Each config takes ~1.5-2.5 h on CPU (sudoku is cheaper: 15 epochs / 6 seeds). Runs serially so the
# per-config parallel_workers do not oversubscribe the machine.
#
# Usage:  bash scratchpad/m33/run_remaining.sh
# Logs:   scratchpad/m33/logs/<config>.log   (per config)   +   this script's own stdout
set -u
cd "$(dirname "$0")/../.." || exit 1
mkdir -p scratchpad/m33/logs

# Full ordered list (same as run_all.sh); the skip check makes already-done ones no-ops.
CONFIGS=(
  m33_converge_w24 m33_converge_w32 m33_converge_w48
  m33_hopfield_w24 m33_hopfield_w32
  m33_mixed_converge_w24 m33_mixed_converge_w32
  m33_disruption_w24 m33_disruption_w32
  m33_sudoku
)

for c in "${CONFIGS[@]}"; do
  # Skip if a completed deltas file already exists for this config (portable glob-match test).
  if ls results/${c}_*_deltas.csv > /dev/null 2>&1; then
    echo "=== $(date +%H:%M:%S) SKIP  $c (already has results) ==="
    continue
  fi
  echo "=== $(date +%H:%M:%S) START $c ==="
  uv run python -m looptab.run --config "configs/experiments/${c}.yaml" \
    > "scratchpad/m33/logs/${c}.log" 2>&1
  rc=$?
  echo "=== $(date +%H:%M:%S) DONE  $c (rc=$rc) ==="
  tail -3 "scratchpad/m33/logs/${c}.log"
  if [ "$rc" -ne 0 ]; then
    echo "!!! $c FAILED (rc=$rc) — see scratchpad/m33/logs/${c}.log; continuing with the rest."
  fi
done
echo "=== $(date +%H:%M:%S) M33 REMAINING RUNS COMPLETE ==="
echo "Completed configs:"
ls results/m33_*_deltas.csv 2>/dev/null | sed 's#results/##;s/_[0-9]*T[0-9]*_deltas.csv//' | sort -u
