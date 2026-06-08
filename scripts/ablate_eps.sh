#!/usr/bin/env bash
# ablate_eps.sh — Phase 1c: how sensitive is the result to ε?
#
# ε is the entropic temperature inside Sinkhorn. Lower ε → sharper transport,
# closer to the unregularised assignment but harder to optimise. Higher ε →
# smoother, more like softmax. The learnable `scale` head already absorbs most
# sharpness, so ε ∈ [0.5, 5] should all behave reasonably; a flat curve here
# is the result we want.
#
# Fixed: routing=balanced_ot, N=512, iters=8, lambda=0.013.
# Varying: ot-eps ∈ {0.5, 1.0, 2.0, 5.0}.
#
# Edit the three paths, then: bash scripts/ablate_eps.sh
set -euo pipefail

DATASET=/path/to/openimages
CKPT=/path/to/dcae_lambda0.013.pth.tar
OUT=./results/ablate_eps

LAMBDA=0.013
EPOCHS=20
BS=16
LR=1e-4
GPUS=8

mkdir -p "$OUT"

for EPS in 0.5 1.0 2.0 5.0; do
  TAG="balanced_ot_eps${EPS}_N512"
  echo "=== finetuning $TAG ==="
  torchrun --nproc_per_node=$GPUS train.py \
    --dataset      "$DATASET" \
    --checkpoint   "$CKPT" \
    --routing      balanced_ot \
    --dict-num     512 \
    --ot-iters     8 \
    --ot-eps       "$EPS" \
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

# Readout: ideally PSNR(ε=0.5..5) within ~0.05 dB of each other AND util_frac
# stays >~0.9 everywhere. That demonstrates the result is not a tuning artifact.
# If ε=0.5 destabilises training (NaN, grad explode), report it — it is the
# expected failure of log-domain Sinkhorn at very small ε.
