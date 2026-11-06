#!/usr/bin/env bash
# Reproduce every experiment. Phase 1: main runs; phase 2: robustness / ablations;
# phase 3: PI-CAX on the same incidents, alarms and grids (needs phases 1-2).
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
picax() { python3 scripts/run_picax.py "$@" >> results/logs/picax.log 2>&1; }
export -f picax
phase3() {
  phase1 | sed 's/ --extras//'
  for det in linear mlp cnn gru; do for ds in sim hai; do echo "--dataset $ds --detector $det --seed 0 --ablation"; done; done
  phase2 | sed 's/ --extras//'
  for ds in sim hai; do for nz in 0.1 0.3; do echo "--dataset $ds --detector cnn --seed 0 --noise $nz --recal --frozen-stats"; done; done
}
case "${1:-all}" in
  1) phase1 | xargs -P "$P" -I{} bash -c 'run {}' ;;
  2) phase2 | xargs -P "$P" -I{} bash -c 'run {}' ;;
  3) phase3 | xargs -P "$P" -I{} bash -c 'picax {}' ;;
  all) phase1 | xargs -P "$P" -I{} bash -c 'run {}'; phase2 | xargs -P "$P" -I{} bash -c 'run {}'
       phase3 | xargs -P "$P" -I{} bash -c 'picax {}' ;;
esac
if [ "${1:-all}" = "all" ]; then
  python3 scripts/run_leakage.py
  python3 scripts/analyze.py --datasets sim hai sim-decoy
  python3 scripts/analyze.py --datasets sim hai --slowdown 10
  python3 scripts/analyze.py --datasets sim hai --slowdown 100
  python3 scripts/analyze_phase2.py
  python3 scripts/analyze_integration.py
  python3 scripts/make_numbers.py && python3 scripts/make_figures.py
fi
