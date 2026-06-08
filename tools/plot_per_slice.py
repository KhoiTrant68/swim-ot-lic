"""plot_per_slice.py — Figure 2: which slices collapse?

DCAE has num_slices cross-attention dictionary modules (one per latent slice).
Collapse is usually not uniform across them — the first slices, which carry
the most rate, often hit it hardest. This plot exposes that.

Grouped bar chart per slice index, softmax vs balanced_ot, at a chosen N.
One subplot per N if multiple are available.

Example:
  python tools/plot_per_slice.py --root results/scan_N \
      --Ns 128 512 2048 --out-pdf results/figures/per_slice.pdf
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 11, "figure.dpi": 120, "savefig.bbox": "tight"})

COLOR = {"softmax": "#d62728", "balanced_ot": "#1f77b4"}
LABEL = {"softmax": "softmax", "balanced_ot": "balanced OT"}


def load_indexed(root: Path):
    """{(routing, N): {slice: util_frac}}"""
    out = {}
    for js in root.glob("*/diag/metrics.json"):
        d = json.loads(js.read_text())
        cfg = d["config"]
        key = (cfg["routing"], cfg["dict_num"])
        out[key] = {s["slice"]: s["util_frac"] for s in d["per_slice"]}
    return out


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--Ns", type=int, nargs="+", default=None,
                   help="if omitted, plot every N seen in the dumps.")
    p.add_argument("--out-pdf", dest="out_pdf", required=True)
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    idx = load_indexed(Path(args.root))
    if not idx:
        print(f"no metrics.json under {args.root}")
        sys.exit(1)

    Ns = sorted(args.Ns) if args.Ns else sorted({n for _, n in idx})
    n_panels = len(Ns)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 3.4), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, N in zip(axes, Ns):
        # collect slice indices from any routing present at this N
        slices = sorted({s for r in ("softmax", "balanced_ot") if (r, N) in idx
                         for s in idx[(r, N)]})
        if not slices:
            continue
        x = np.arange(len(slices))
        w = 0.4
        for i, r in enumerate(("softmax", "balanced_ot")):
            if (r, N) not in idx:
                continue
            y = [idx[(r, N)].get(s, 0.0) for s in slices]
            ax.bar(x + (i - 0.5) * w, y, w, color=COLOR[r], label=LABEL[r])
        ax.set_xticks(x)
        ax.set_xticklabels(slices)
        ax.set_xlabel("slice index")
        ax.set_title(f"N = {N}")
        ax.set_ylim(0, 1.02)
        ax.axhline(1.0, color="gray", lw=0.6, ls=":")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("util_frac")
    axes[-1].legend(loc="lower right", frameon=False)

    out = Path(args.out_pdf)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"wrote {out} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main(sys.argv[1:])
