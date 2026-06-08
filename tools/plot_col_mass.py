"""plot_col_mass.py — Figure 3: the column-mass distribution.

This is the single most convincing figure for the paper. For each atom (sorted
by usage) we plot its average column mass on log-log axes.

  softmax at large N:  steep curve, a few atoms own most of the mass, a long
                       tail of dead atoms below the 1/N reference line.
  balanced_ot:         flat near 1/N across (almost) all atoms — what the
                       theory predicts.

By default we use the slice with the largest gap between the two routings; that
is the slice where the story is clearest. Pass --slice to override.

Example:
  python tools/plot_col_mass.py --root results/scan_N --N 512 \
      --out-pdf results/figures/col_mass_N512.pdf
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


def find_slice_files(root: Path, N: int):
    """Return {routing: metrics_dict} for the given N (one file per routing)."""
    out = {}
    for js in root.glob("*/diag/metrics.json"):
        d = json.loads(js.read_text())
        if d["config"]["dict_num"] == N:
            out[d["config"]["routing"]] = d
    return out


def pick_slice(diag_by_routing, override=None):
    """Pick the slice index with the largest softmax-vs-OT gap in util_frac,
    falling back to slice 0 if both routings are not present."""
    if override is not None:
        return override
    if "softmax" not in diag_by_routing or "balanced_ot" not in diag_by_routing:
        return 0
    s_soft = {s["slice"]: s["util_frac"] for s in diag_by_routing["softmax"]["per_slice"]}
    s_ot   = {s["slice"]: s["util_frac"] for s in diag_by_routing["balanced_ot"]["per_slice"]}
    gaps = {i: s_ot[i] - s_soft.get(i, 0.0) for i in s_ot}
    return max(gaps, key=gaps.get)


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--slice", type=int, default=None,
                   help="slice index to plot; default = the most informative one.")
    p.add_argument("--out-pdf", dest="out_pdf", required=True)
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    diag = find_slice_files(Path(args.root), args.N)
    if not diag:
        print(f"no metrics.json with N={args.N} under {args.root}")
        sys.exit(1)
    sl = pick_slice(diag, args.slice)

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for routing, d in diag.items():
        sd = next((s for s in d["per_slice"] if s["slice"] == sl), None)
        if sd is None:
            continue
        cm = np.asarray(sd["col_mass_sorted"])
        x = np.arange(1, len(cm) + 1)
        ax.plot(x, cm + 1e-12, color=COLOR.get(routing, "k"),
                label=f"{LABEL.get(routing, routing)} (util={sd['util_frac']:.2f})")
    ax.axhline(1.0 / args.N, color="gray", lw=0.7, ls="--",
               label=f"uniform 1/N = {1.0/args.N:.1e}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("atom rank (sorted by mass)")
    ax.set_ylabel("mean column mass")
    ax.set_title(f"column-mass distribution  (N={args.N}, slice {sl})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", frameon=False)

    out = Path(args.out_pdf)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"wrote {out} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main(sys.argv[1:])
