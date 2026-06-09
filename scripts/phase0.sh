#!/usr/bin/env bash
# phase0.sh — sanity check BEFORE running scan_N.sh.
#
# Purpose: prove that our harness (model + finetune + eval) can reproduce the
# published DCAE number at lambda=0.013 using the baseline cell
# (routing=softmax, N=128). If this number does NOT match DCAE within
# ~0.1 dB / ~3% bpp, STOP — every Phase-1 result will be meaningless because
# the baseline is wrong. Do NOT touch routing until this is green.
#
# Why softmax/N=128 specifically: this is the cell that should behave
# identically to vanilla DCAE (softmax = the published routing, N=128 = the
# published dict size). Any deviation here is a harness bug, not a routing
# effect.
#
# Edit the four paths, then: bash scripts/phase0.sh
set -euo pipefail

DATASET=/kaggle/input/datasets/tranjohan/data-1000-test/dataset_1000_test          # train/ + test/ subfolders (compressai ImageFolder)
CKPT=./checkpoints/dcae_lambda0.013.pth.tar
KODAK=/kaggle/input/datasets/khitrnminh/kodak-test               # 24 PNGs
OUT=./results/phase0

LAMBDA=0.013
EPOCHS=10
BS=4
LR=1e-4
GPUS=2

TAG="softmax_N128"
RUN_DIR="$OUT/$TAG"
CKPT_OUT="$RUN_DIR/$LAMBDA/checkpoint_best.pth.tar"

mkdir -p "$OUT"

# ── 1. Finetune the baseline cell ─────────────────────────────────────────────
echo "=== finetuning $TAG (lambda=$LAMBDA) ==="
torchrun --nproc_per_node=$GPUS train.py \
  --dataset      "$DATASET" \
  --checkpoint   "$CKPT" \
  --routing      softmax \
  --dict-num     128 \
  --lambda       "$LAMBDA" \
  --epochs       "$EPOCHS" \
  --batch-size   "$BS" \
  --learning-rate "$LR" \
  --type         mse \
  --cuda \
  --save_path    "$RUN_DIR" \
  --finetune

# ── 2. Eval on Kodak (forward-estimate path; --real adds AC round-trip) ───────
echo "=== eval $TAG on Kodak (forward estimate) ==="
python eval.py --cuda \
  --routing    softmax \
  --dict-num   128 \
  --checkpoint "$CKPT_OUT" \
  --data       "$KODAK" \
  | tee "$RUN_DIR/eval_kodak_forward.log"

echo "=== eval $TAG on Kodak (real AC) ==="
python eval.py --cuda --real \
  --routing    softmax \
  --dict-num   128 \
  --checkpoint "$CKPT_OUT" \
  --data       "$KODAK" \
  | tee "$RUN_DIR/eval_kodak_real.log"

# ── Readout ───────────────────────────────────────────────────────────────────
# Compare the printed "Average PSNR / Bit-rate" against DCAE's published Kodak
# point at lambda=0.013. Concretely:
#   PASS  : within ~0.10 dB PSNR AND within ~3% bpp of DCAE published.
#   SOFT-FAIL : within ~0.25 dB but bpp drift > 5% → suspect data pipeline
#               (normalisation, padding, channel order). Fix before Phase 1.
#   HARD-FAIL : > 0.4 dB drift OR PSNR < expected → harness bug; do NOT proceed.
#
# Also check the forward-estimate vs real numbers should be within ~0.01 bpp —
# if not, the entropy-coder side of compress/decompress is misconfigured.
echo "Phase 0 done. Compare $RUN_DIR/eval_kodak_*.log against DCAE published."
