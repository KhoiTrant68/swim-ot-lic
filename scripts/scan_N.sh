#!/usr/bin/env bash
# scan_N.sh — Phase 1: the make-or-break scaling sweep.
#
# Hypothesis under test:
#   softmax loses utilisation (dead atoms) as N grows; balanced OT holds it,
#   and OT's RD point keeps improving with N while softmax plateaus/worsens.
#
# Design: FINETUNE from the DCAE checkpoint at ONE mid-rate lambda for every
# (routing x N) cell. Cheap on 8xH100 (no from-scratch training).
#
# Edit the three paths, then: bash scan_N.sh
set -euo pipefail

DATASET=/path/to/openimages          # has train/ and test/ subfolders (compressai ImageFolder)
CKPT=/path/to/dcae_lambda0.013.pth.tar
OUT=./results/scan_N

LAMBDA=0.013                         # single mid-rate point for the sweep
EPOCHS=40                            # short finetune; bump if loss still moving
BS=16                                # per-GPU; 8 GPUs -> global batch 128
LR=1e-4                              # fresh LR (do NOT resume DCAE optimizer)
GPUS=8

mkdir -p "$OUT"

for ROUTING in softmax balanced_ot; do
  for N in 128 512 2048; do          # add 256/1024 once 128/512/2048 look sane
    TAG="${ROUTING}_N${N}"
    echo "=== finetuning $TAG (lambda=$LAMBDA) ==="
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
      --save_path    "$OUT/$TAG" \
      --no-continue-train            # fresh optimizer/scheduler; load weights only
    echo "=== done $TAG ==="
  done
done

# Win/lose readout (Phase 1):
#   1. SANITY: softmax_N128 must match published DCAE at lambda=0.013.
#      If it does not, STOP — the harness is wrong, not the routing.
#   2. Plot util_frac & col_entropy (sinkhorn.plan_utilisation on a few Kodak
#      images) vs N, for softmax vs balanced_ot.
#   3. Plot the converged RD point (bpp, PSNR) and loss = lambda*255^2*MSE + bpp
#      vs N for each routing.
#
#   OT EARNS ITS PLACE iff at some N* > 128:
#       balanced_ot keeps util_frac high (>~0.9) while softmax drops, AND
#       balanced_ot's RD point beats the best softmax cell.
#   If softmax already keeps util ~1 at every N you test and RD does not improve
#   with N, OT buys nothing here — report that honestly and stop.
#
# Phase 2 (only the winning routing+N*): finetune the full lambda grid
#   {0.0018, 0.0035, 0.0067, 0.013, 0.025, 0.05}, run eval.py on
#   Kodak/Tecnick/CLIC, compute BD-rate vs VTM and vs DCAE.