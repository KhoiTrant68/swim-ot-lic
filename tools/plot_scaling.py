"""plot_scaling.py — Figure 1, the make-or-break plot.

Walks <root>/*/diag/metrics.json (typically scan_N output) and produces a
two-panel figure:
    (a) util_frac vs N  — softmax should drop, balanced_ot should stay near 1
    (b) PSNR     vs N  — balanced_ot should keep improving where softmax flatlines

If the hypothesis fails (softmax holds util at every N, OT does not pull ahead
on PSNR) this plot will say so honestly — read it that way.

Example:
  python tools/plot_scaling.py --root results/scan_N --out-pdf results/figures/scaling.pdf
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "figure.dpi": 120, "savefig.bbox": "tight"})

ROUTING_STYLE = {
    "softmax":     {"color": "#d62728", "marker": "o", "label": "softmax (DCAE)"},
    "balanced_ot": {"color": "#1f77b4", "marker": "s", "label": "balanced OT (ours)"},
}


def load_runs(root: Path):
    """Returns {routing: sorted [(N, util_frac, psnr, bpp), ...]} from all metrics.json under root."""
    runs = defaultdict(list)
    for js in root.glob("*/diag/metrics.json"):
        d = json.loads(js.read_text())
        cfg = d["config"]
        runs[cfg["routing"]].append(
            (cfg["dict_num"], d["util_frac"], d["psnr"], d["bpp"])
        )
    return {r: sorted(set(v)) for r, v in runs.items()}


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="results dir holding <run>/diag/metrics.json")
    p.add_argument("--out-pdf", dest="out_pdf", required=True)
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    runs = load_runs(Path(args.root))
    if not runs:
        print(f"no metrics.json under {args.root} — did you run dump_all.sh?")
        sys.exit(1)

    fig, (axU, axP) = plt.subplots(1, 2, figsize=(9.0, 3.6), sharex=True)
    for routing, pts in runs.items():
        sty = ROUTING_STYLE.get(routing, {"label": routing})
        Ns = [n for n, _, _, _ in pts]
        u  = [v for _, v, _, _ in pts]
        ps = [v for _, _, v, _ in pts]
        axU.plot(Ns, u, **sty)
        axP.plot(Ns, ps, **sty)

    for ax in (axU, axP):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("codebook size N")
        ax.grid(True, alpha=0.3)
    axU.set_ylabel("util_frac  (fraction of atoms alive)")
    axU.set_ylim(0, 1.02)
    axU.axhline(1.0, color="gray", lw=0.6, ls=":")
    axU.set_title("(a) codebook utilisation vs N")
    axP.set_ylabel("PSNR  [dB]")
    axP.set_title("(b) rate–distortion (PSNR) vs N")
    axP.legend(loc="best", frameon=False)

    out = Path(args.out_pdf)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"wrote {out} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main(sys.argv[1:])
