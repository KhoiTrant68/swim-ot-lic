"""dump_diagnostics.py — forward-eval one checkpoint on a folder of images and
dump everything the Phase-1 figures need into a single JSON.

Output (in --out-dir):
  metrics.json  — config + aggregate {bpp, psnr, util_frac, col_entropy}
                  + per-slice {util_frac, col_entropy, col_mass_sorted (len N)}
                  + per-image {name, bpp, psnr}.

The col_mass_sorted field is the killer plot: it shows softmax's long tail
(few atoms dominate, most dead) vs balanced OT's near-flat distribution.

Example:
  python tools/dump_diagnostics.py --cuda \
      --checkpoint results/scan_N/balanced_ot_N512/0.013/checkpoint_best.pth.tar \
      --data /path/to/kodak \
      --out-dir results/scan_N/balanced_ot_N512/diag
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# allow `python tools/dump_diagnostics.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.dcae_ot import DCAE_OT
from modules.sinkhorn import plan_utilisation


def compute_psnr(a, b):
    return -10 * math.log10(torch.mean((a - b) ** 2).item())


def compute_bpp(out_net):
    s = out_net["x_hat"].size()
    num_pixels = s[0] * s[2] * s[3]
    return sum(torch.log(lk).sum() / (-math.log(2) * num_pixels)
               for lk in out_net["likelihoods"].values()).item()


def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    left = (new_w - w) // 2
    right = new_w - w - left
    top = (new_h - h) // 2
    bottom = new_h - h - top
    return F.pad(x, (left, right, top, bottom), mode="constant", value=0), (left, right, top, bottom)


def crop(x, padding):
    return F.pad(x, (-padding[0], -padding[1], -padding[2], -padding[3]))


def col_mass_sorted(probs: torch.Tensor) -> torch.Tensor:
    """Mean column mass over (B, heads, HW), normalised, sorted descending."""
    N = probs.shape[-1]
    flat = probs.reshape(-1, N)
    cm = flat.mean(0)
    cm = cm / (cm.sum() + 1e-12)
    return torch.sort(cm, descending=True).values


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--out-dir", dest="out_dir", type=str, required=True)
    p.add_argument("--max-images", dest="max_images", type=int, default=24)
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    device = "cuda:0" if args.cuda else "cpu"
    p = 128

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = {k: ckpt.get(k) for k in ("routing", "dict_num", "ot_iters", "ot_eps", "lambda")}
    cfg["routing"] = cfg["routing"] or "softmax"
    cfg["dict_num"] = cfg["dict_num"] or 128
    cfg["ot_iters"] = cfg["ot_iters"] or 8
    cfg["ot_eps"] = cfg["ot_eps"] or 1.0
    print("config:", cfg)

    net = DCAE_OT(routing=cfg["routing"], dict_num=cfg["dict_num"],
                  ot_iters=cfg["ot_iters"], ot_eps=cfg["ot_eps"]).to(device).eval()
    sd = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    net.load_state_dict(sd)

    img_list = sorted(f for f in os.listdir(args.data)
                      if f.lower().endswith((".png", ".jpg", ".jpeg")))
    img_list = img_list[:args.max_images]

    n_slices = len(net.dt_cross_attention)
    N = cfg["dict_num"]
    col_mass_sum = [torch.zeros(N) for _ in range(n_slices)]
    util_sum = [0.0] * n_slices
    ent_sum = [0.0] * n_slices
    PSNR = BPP = 0.0
    per_image = []

    for name in img_list:
        img = transforms.ToTensor()(
            Image.open(os.path.join(args.data, name)).convert("RGB")
        ).to(device).unsqueeze(0)
        x_padded, padding = pad(img, p)
        with torch.no_grad():
            out_net = net.forward(x_padded)
            out_net["x_hat"].clamp_(0, 1)
            x_hat = crop(out_net["x_hat"], padding)
            psnr_i = compute_psnr(img, x_hat)
            bpp_i = compute_bpp(out_net)
            PSNR += psnr_i
            BPP += bpp_i
            for i, a in enumerate(net.dt_cross_attention):
                probs = a.last_probs.cpu()
                col_mass_sum[i] += col_mass_sorted(probs)
                s = plan_utilisation(probs)
                util_sum[i] += s["util_frac"]
                ent_sum[i] += s["col_entropy"]
        per_image.append({"name": name, "bpp": bpp_i, "psnr": psnr_i})
        print(f"  {name}: bpp {bpp_i:.4f} psnr {psnr_i:.2f}")

    n_img = len(img_list)
    diag = {
        "config": cfg,
        "checkpoint": args.checkpoint,
        "n_images": n_img,
        "bpp": BPP / n_img,
        "psnr": PSNR / n_img,
        "util_frac": sum(util_sum) / (n_img * n_slices),
        "col_entropy": sum(ent_sum) / (n_img * n_slices),
        "per_slice": [
            {"slice": i,
             "util_frac": util_sum[i] / n_img,
             "col_entropy": ent_sum[i] / n_img,
             "col_mass_sorted": (col_mass_sum[i] / n_img).tolist()}
            for i in range(n_slices)
        ],
        "per_image": per_image,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "metrics.json"
    out.write_text(json.dumps(diag, indent=2))
    print(f"wrote {out}")
    print(f"agg: bpp {diag['bpp']:.4f} psnr {diag['psnr']:.2f} "
          f"util {diag['util_frac']:.3f} ent {diag['col_entropy']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
