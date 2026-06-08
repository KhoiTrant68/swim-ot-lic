#!/usr/bin/env bash
# plot_all.sh — generate the Phase-1 figures from dumped diagnostics.
#
# Assumes dump_all.sh has already been run, so each <run>/diag/metrics.json
# exists. The eps/iters figure needs its own ablate roots and is optional.
#
# Usage:
#   bash scripts/plot_all.sh <scan_N_root> [eps_iters: <iters_root> <eps_root>]
# Example:
#   bash scripts/plot_all.sh results/scan_N \
#       results/ablate_iters results/ablate_eps
set -euo pipefail

SCAN_ROOT="${1:?usage: plot_all.sh <scan_N_root> [<iters_root> <eps_root>]}"
ITERS_ROOT="${2:-}"
EPS_ROOT="${3:-}"
FIG_DIR="${SCAN_ROOT}/figures"
mkdir -p "$FIG_DIR"

echo "=== Figure 1: scaling ==="
python tools/plot_scaling.py    --root "$SCAN_ROOT" \
                                --out-pdf "$FIG_DIR/scaling.pdf"

echo "=== Figure 2: per-slice utilisation ==="
python tools/plot_per_slice.py  --root "$SCAN_ROOT" \
                                --out-pdf "$FIG_DIR/per_slice.pdf"

echo "=== Figure 3: column-mass distribution (N=512) ==="
python tools/plot_col_mass.py   --root "$SCAN_ROOT" --N 512 \
                                --out-pdf "$FIG_DIR/col_mass_N512.pdf"

if [[ -n "$ITERS_ROOT" && -n "$EPS_ROOT" ]]; then
  echo "=== Figure 4: iters & eps sensitivity ==="
  python tools/plot_eps_iters.py  --iters-root "$ITERS_ROOT" \
                                  --eps-root   "$EPS_ROOT" \
                                  --out-pdf "$FIG_DIR/eps_iters.pdf"
else
  echo "(skipping Figure 4: pass <iters_root> <eps_root> to include it)"
fi

echo "all figures in: $FIG_DIR"
