#!/usr/bin/env bash
# Run all 10 M33 decomposition configs sequentially (each uses parallel_workers, so serial across
# configs avoids CPU oversubscription). Logs per config; prints the deltas file path on completion.
# cd to repo root via this script's location (portable, per M32 review).
set -u
cd "$(dirname "$0")/../.." || exit 1
mkdir -p scratchpad/m33/logs
CONFIGS=(
  m33_converge_w24 m33_converge_w32 m33_converge_w48
  m33_hopfield_w24 m33_hopfield_w32
  m33_mixed_converge_w24 m33_mixed_converge_w32
  m33_disruption_w24 m33_disruption_w32
  m33_sudoku
)
for c in "${CONFIGS[@]}"; do
  echo "=== $(date +%H:%M:%S) START $c ==="
  uv run python -m looptab.run --config "configs/experiments/${c}.yaml" \
    > "scratchpad/m33/logs/${c}.log" 2>&1
  rc=$?
  echo "=== $(date +%H:%M:%S) DONE  $c (rc=$rc) ==="
  tail -3 "scratchpad/m33/logs/${c}.log"
done
echo "=== ALL M33 RUNS COMPLETE ==="
