#!/bin/bash
set -e
cd /Users/mattsimmons/looptab
for cfg in etth1_h192 etth1_h336 weather_h192 weather_h336; do
  echo "=== START $cfg $(date) ==="
  OMP_NUM_THREADS=1 uv run python -m looptab.run --config configs/experiments/m32_${cfg}.yaml > scratchpad/m32/${cfg}.log 2>&1
  echo "=== DONE $cfg exit=$? $(date) ==="
done
echo "ALL DONE"
