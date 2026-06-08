"""plot_eps_iters.py — Figure 4: iters and eps sensitivity.

Reviewers will ask:
  - "is 8 Sinkhorn iters arbitrary?"  → show PSNR saturates by ~8.
  - "is the result fragile to eps?"   → show PSNR is flat across eps ∈ [0.5, 5].

Reads two roots, one per ablation. Each <root>/<run>/diag/metrics.json must
have its routing/dict_num/ot_iters/ot_eps set (the trainer saves these in the
checkpoint and dump_diagnostics copies them into "config").

Example:
  python tools/plot_eps_iters.py \
      --iters-root results/ablate_iters \
      --eps-root   results/ablate_eps \
      --out-pdf results/figures/eps_iters.pdf
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "figure.dpi": 120, "savefig.bbox": "tight"})


def load_axis(root: Path, key: str):
    """Returns sorted [(x, psnr, bpp, util_frac), ...] where x is cfg[key]."""
    pts = []
    for js in root.glob("*/diag/metrics.json"):
        d = json.loads(js.read_text())
        cfg = d["config"]
        pts.append((cfg[key], d["psnr"], d["bpp"], d["util_frac"]))
    return sorted(set(pts))


def panel(ax, pts, xlabel, title, logx=False):
    if not pts:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
        ax.set_title(title)
        return
    xs = [p[0] for p in pts]
    ps = [p[1] for p in pts]
    us = [p[3] for p in pts]
    ax.plot(xs, ps, marker="s", color="#1f77b4", label="PSNR")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("PSNR [dB]", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    if logx:
        ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax2 = ax.twinx()
    ax2.plot(xs, us, marker="o", color="#2ca02c", linestyle="--", label="util_frac")
    ax2.set_ylabel("util_frac", color="#2ca02c")
    ax2.set_ylim(0, 1.02)
    ax2.tick_params(axis="y", labelcolor="#2ca02c")


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--iters-root", dest="iters_root", required=True)
    p.add_argument("--eps-root",   dest="eps_root",   required=True)
    p.add_argument("--out-pdf",    dest="out_pdf",    required=True)
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    pts_i = load_axis(Path(args.iters_root), "ot_iters")
    pts_e = load_axis(Path(args.eps_root),   "ot_eps")
    fig, (axI, axE) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    panel(axI, pts_i, "Sinkhorn iters", "(a) iters saturation (eps=1, N=512)", logx=True)
    panel(axE, pts_e, "ε (entropic temperature)",
          "(b) eps sensitivity (iters=8, N=512)", logx=True)
    out = Path(args.out_pdf)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"wrote {out} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main(sys.argv[1:])
