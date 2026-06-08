#!/usr/bin/env bash
# dump_all.sh — run forward-eval diagnostics on every checkpoint_best.pth.tar
# under a results root. Each run becomes <run>/diag/metrics.json, which the
# plot_* tools consume.
#
# Layout expected (matches train.py's save_path/<lambda>/checkpoint_best.pth.tar):
#   <results_root>/<run_tag>/<lambda>/checkpoint_best.pth.tar
#
# Usage:
#   bash scripts/dump_all.sh <results_root> <eval_image_dir>
# Example:
#   bash scripts/dump_all.sh results/scan_N /datasets/kodak
set -euo pipefail

ROOT="${1:?usage: dump_all.sh <results_root> <eval_image_dir>}"
DATA="${2:?usage: dump_all.sh <results_root> <eval_image_dir>}"

shopt -s nullglob
found=0
for ckpt in "$ROOT"/*/*/checkpoint_best.pth.tar; do
  found=1
  rundir="$(dirname "$(dirname "$ckpt")")"
  outdir="$rundir/diag"
  if [[ -f "$outdir/metrics.json" ]]; then
    echo "=== skip $rundir (metrics.json exists) ==="
    continue
  fi
  echo "=== dumping $rundir ==="
  python tools/dump_diagnostics.py --cuda \
    --checkpoint "$ckpt" \
    --data       "$DATA" \
    --out-dir    "$outdir"
done
if [[ $found -eq 0 ]]; then
  echo "no checkpoint_best.pth.tar found under $ROOT — wrong path?" >&2
  exit 1
fi
