#!/usr/bin/env bash
# Reproduce every experiment. Phase 1: main runs; phase 2: robustness / ablations.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs
P=${PARALLEL:-4}
run() { python3 scripts/run_localization.py "$@" > "results/logs/$(echo "$@" | tr ' -' '__').log" 2>&1; }
export -f run
phase1() {
  for ds in sim hai; do for det in linear mlp cnn gru; do for s in 0 1 2; do
    echo "--dataset $ds --detector $det --seed $s --extras"; done; done; done
}
phase2() {
  for det in linear cnn; do for s in 0 1 2; do echo "--dataset sim-decoy --detector $det --seed $s"; done; done
  for ds in sim hai; do for nz in 0.1 0.3; do echo "--dataset $ds --detector cnn --seed 0 --noise $nz"
    echo "--dataset $ds --detector cnn --seed 0 --noise $nz --recal --extras"; done
    for rg in 32 120 300; do echo "--dataset $ds --detector cnn --seed 0 --ref-gap $rg"; done; done
}
case "${1:-all}" in
  1) phase1 | xargs -P "$P" -I{} bash -c 'run {}' ;;
  2) phase2 | xargs -P "$P" -I{} bash -c 'run {}' ;;
  all) phase1 | xargs -P "$P" -I{} bash -c 'run {}'; phase2 | xargs -P "$P" -I{} bash -c 'run {}' ;;
esac
if [ "${1:-all}" = "all" ]; then
  python3 scripts/run_leakage.py
  python3 scripts/analyze.py --datasets sim hai sim-decoy
  python3 scripts/analyze_phase2.py
fi
