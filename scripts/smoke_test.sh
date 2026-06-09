#!/usr/bin/env bash
# smoke_test.sh — derisk the CODE on Kaggle 2xGPU. NOT a science test.
#
# Goal: prove the harness end-to-end works for BOTH routings before burning
# real cloud compute on Phase 0/1. We check:
#   * model builds + finetune runs + checkpoint saves (both routings)
#   * eval.py runs and reports bpp/PSNR/util_frac
#   * balanced_ot branch produces non-trivial util_frac (close to 1.0 at N=128)
#   * real AC bpp ~= forward-estimate bpp (entropy coder configured)
#
# Do NOT read the PSNR numbers as evidence of anything scientific — 1000 imgs
# * 10 epochs is far below what's needed to converge or reproduce DCAE.
# A PSNR drift of 1-2 dB from DCAE published is EXPECTED here.
#
# Edit the four paths, then: bash scripts/smoke_test.sh
set -euo pipefail

DATASET=/kaggle/input/datasets/tranjohan/data-1000-test/dataset_1000_test
CKPT=/kaggle/input/datasets/khitrnminh/dcae-0-013-checkpoint-09062026/0.013checkpoint_best.pth.tar
KODAK=/kaggle/input/datasets/khitrnminh/kodak-test
OUT=./results/smoke

LAMBDA=0.013
EPOCHS=10
BS=4
LR=1e-4
GPUS=2

mkdir -p "$OUT"

run_cell () {
  local ROUTING=$1
  local N=$2
  local TAG="${ROUTING}_N${N}"
  local RUN_DIR="$OUT/$TAG"
  local CKPT_OUT="$RUN_DIR/$LAMBDA/checkpoint_best.pth.tar"

  echo "============================================================"
  echo "  finetune  $TAG"
  echo "============================================================"
  torchrun --nproc_per_node=$GPUS train.py \
    --dataset      "$DATASET" \
    --checkpoint   "$CKPT" \
    --routing      "$ROUTING" \
    --dict-num     "$N" \
    --ot-iters     8 \
    --ot-eps       1.0 \
    --lambda       "$LAMBDA" \
    --epochs       "$EPOCHS" \
    --batch-size   "$BS" \
    --learning-rate "$LR" \
    --type         mse \
    --cuda \
    --save_path    "$RUN_DIR" \
    --finetune

  echo "------------------------------------------------------------"
  echo "  eval $TAG on Kodak (forward estimate)"
  echo "------------------------------------------------------------"
  python eval.py --cuda \
    --routing    "$ROUTING" \
    --dict-num   "$N" \
    --checkpoint "$CKPT_OUT" \
    --data       "$KODAK" \
    | tee "$RUN_DIR/eval_kodak_forward.log"

  # only run --real on the OT cell to keep wall time short; AC behaves the same
  if [[ "$ROUTING" == "balanced_ot" ]]; then
    echo "------------------------------------------------------------"
    echo "  eval $TAG on Kodak (real AC roundtrip)"
    echo "------------------------------------------------------------"
    python eval.py --cuda --real \
      --routing    "$ROUTING" \
      --dict-num   "$N" \
      --checkpoint "$CKPT_OUT" \
      --data       "$KODAK" \
      | tee "$RUN_DIR/eval_kodak_real.log"
  fi
}

run_cell softmax     128
run_cell balanced_ot 128

# ── Readout: green-light checklist ────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  SMOKE TEST READOUT — manually check $OUT/*/eval_kodak_*.log"
echo "============================================================"
echo "  PASS criteria (all four must hold):"
echo "    1. Both finetunes ran to completion, checkpoint_best.pth.tar exists."
echo "    2. Eval prints non-NaN bpp/PSNR for every Kodak image."
echo "    3. balanced_ot eval prints 'Codebook util_frac' line with value ~> 0.9."
echo "       (softmax may or may not — doesn't matter at N=128.)"
echo "    4. balanced_ot real-AC bpp is within ~0.02 of its forward-estimate bpp."
echo ""
echo "  Anything else is a CODE bug to fix BEFORE requesting cloud compute."
echo "  Do NOT interpret absolute PSNR — 10 epochs on 1000 imgs is too small."
