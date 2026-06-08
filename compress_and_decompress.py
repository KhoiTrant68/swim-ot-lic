"""
compress_and_decompress.py — single-image round-trip demo for DCAE_OT.

Encodes one image to a real bitstream, decodes it back, reports true bpp / PSNR
/ MS-SSIM, and optionally saves the reconstruction. Useful as a quick sanity
check that compress()/decompress() are byte-consistent under OT routing
(Sinkhorn is deterministic, so they are).

Example:
  python compress_and_decompress.py --cuda --routing balanced_ot --dict-num 512 \
      --checkpoint .../checkpoint_best.pth.tar --input kodim01.png --output rec.png
"""

import argparse
import math
import sys
import warnings

import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms

from models.dcae_ot import DCAE_OT

warnings.filterwarnings("ignore")


def compute_psnr(a, b):
    return -10 * math.log10(torch.mean((a - b) ** 2).item())


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


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, default=None, help="Save reconstruction here.")
    p.add_argument("--routing", type=str, default="softmax", choices=["softmax", "balanced_ot"])
    p.add_argument("--dict-num", dest="dict_num", type=int, default=128)
    p.add_argument("--ot-iters", dest="ot_iters", type=int, default=8)
    p.add_argument("--ot-eps", dest="ot_eps", type=float, default=1.0)
    return p.parse_args(argv)


def main(argv):
    torch.backends.cudnn.enabled = False
    args = parse_args(argv)
    device = "cuda:0" if args.cuda else "cpu"
    p = 128

    net = DCAE_OT(routing=args.routing, dict_num=args.dict_num,
                  ot_iters=args.ot_iters, ot_eps=args.ot_eps).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    net.load_state_dict(sd)
    net.update(force=True)

    img = transforms.ToTensor()(Image.open(args.input).convert("RGB")).to(device)
    x = img.unsqueeze(0)
    x_padded, padding = pad(x, p)

    with torch.no_grad():
        out_enc = net.compress(x_padded)
        out_dec = net.decompress(out_enc["strings"], out_enc["shape"])

    x_hat = crop(out_dec["x_hat"], padding).clamp_(0, 1)
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
    psnr = compute_psnr(x, x_hat)
    msssim = -10 * math.log10(1 - ms_ssim(x, x_hat, data_range=1.0).item())

    print(f"bpp      : {bpp:.4f}")
    print(f"PSNR     : {psnr:.2f} dB")
    print(f"MS-SSIM  : {msssim:.2f} dB")

    if args.output:
        transforms.ToPILImage()(x_hat.squeeze(0).cpu()).save(args.output)
        print(f"saved reconstruction -> {args.output}")


if __name__ == "__main__":
    main(sys.argv[1:])