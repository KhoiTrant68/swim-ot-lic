#!/usr/bin/env bash
# ablate_iters.sh — Phase 1b: does the Sinkhorn iteration count matter?
#
# Hypothesis: balanced Sinkhorn converges fast at eps=1.0; 4-8 iters is enough,
# and pushing to 16/32 does not move PSNR. We want to defend "n_iters=8" with
# a saturation curve.
#
# Fixed: routing=balanced_ot, N=512, eps=1.0, lambda=0.013.
# Varying: ot-iters ∈ {2, 4, 8, 16, 32}.
#
# Edit the three paths, then: bash scripts/ablate_iters.sh
set -euo pipefail

DATASET=/path/to/openimages
CKPT=/path/to/dcae_lambda0.013.pth.tar
OUT=./results/ablate_iters

LAMBDA=0.013
EPOCHS=20            # short finetune; this ablation is about *plateau*, not RD
BS=16
LR=1e-4
GPUS=8

mkdir -p "$OUT"

for ITERS in 2 4 8 16 32; do
  TAG="balanced_ot_iter${ITERS}_N512"
  echo "=== finetuning $TAG ==="
  torchrun --nproc_per_node=$GPUS train.py \
    --dataset      "$DATASET" \
    --checkpoint   "$CKPT" \
    --routing      balanced_ot \
    --dict-num     512 \
    --ot-iters     "$ITERS" \
    --ot-eps       1.0 \
    --lambda       "$LAMBDA" \
    --epochs       "$EPOCHS" \
    --batch-size   "$BS" \
    --learning-rate "$LR" \
    --type         mse \
    --cuda \
    --save_path    "$OUT/$TAG" \
    --finetune
  echo "=== done $TAG ==="
done

# Readout: PSNR(iters=2) < PSNR(iters=4) ≈ PSNR(iters=8) ≈ PSNR(iters=16) — that's
# the saturation curve. If iters=2 is already plateau, drop the default to 4.
# If PSNR keeps climbing past 16, something is off (eps too small? log-domain
# underflow?) and 8 is too few.
