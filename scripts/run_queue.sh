#!/usr/bin/env bash
# Run a list of run_localization.py argument lines with N parallel workers.
# Usage: bash scripts/run_queue.sh <file-with-arg-lines> [N]
cd "$(dirname "$0")/.."
mkdir -p results/logs
while read -r line; do echo "$line"; done < "$1" | xargs -P "${2:-4}" -I{} bash -c 'python3 scripts/run_localization.py {} > "results/logs/$(echo "{}" | tr " -" "__").log" 2>&1'
