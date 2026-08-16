#!/bin/bash
# Sequential driver for the M34 milestone's full sweeps. Runs on ONE GPU, so these must be
# SEQUENTIAL (not concurrent) -- the electricity smoke test alone used ~93% of the RTX 2070's
# 8GB VRAM, so two concurrent big configs would risk OOM. Logs each config to its own file under
# scratchpad/m34/logs/ and continues past a failure (no `&&`) so one bad config doesn't block the
# rest of the queue.
set -x
cd /c/Users/scifi/looptab
mkdir -p scratchpad/m34/logs

PY=".venv/Scripts/python.exe"

run() {
  name="$1"; cfg="$2"
  if [[ -f "scratchpad/m34/logs/${name}.done" ]]; then
    echo "=== SKIP $name (already completed) $(date) ==="
    return 0
  fi
  echo "=== START $name $(date) ==="
  "$PY" -m looptab.run --config "$cfg" > "scratchpad/m34/logs/${name}.log" 2>&1
  status=$?
  if [[ $status -eq 0 ]]; then
    printf 'completed %s\n' "$(date -Iseconds)" > "scratchpad/m34/logs/${name}.done"
  fi
  echo "=== END $name $(date) exit=$status ==="
}

# Synthetic scale-up (direction A) -- cheaper / more validated first.
run converge_w64   configs/experiments/m34_converge_w64.yaml
run converge_w96   configs/experiments/m34_converge_w96.yaml
run sudoku9        configs/experiments/m34_sudoku9.yaml

# Forecasting breadth (direction B) -- cheap tier (identical geometry to etth1) first.
run etth2_h24       configs/experiments/m34_etth2_h24.yaml
run ettm1_h24       configs/experiments/m34_ettm1_h24.yaml
run ettm2_h24       configs/experiments/m34_ettm2_h24.yaml

# High-channel-count tier -- expensive; electricity confirmed compute-bound (~45-60min/seed in
# smoke testing), traffic untested (wider still -- watch scratchpad/m34/logs/traffic_h24.log for
# an early OOM rather than assuming it'll finish).
run electricity_h24 configs/experiments/m34_electricity_h24.yaml
run traffic_h24      configs/experiments/m34_traffic_h24.yaml

echo "=== ALL M34 RUNS COMPLETE $(date) ==="
