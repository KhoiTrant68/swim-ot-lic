"""
eval.py — evaluate a finetuned DCAE_OT checkpoint on a folder of images.

Changes vs DCAE eval.py:
  * build DCAE_OT(routing=..., dict_num=...) — must match how the checkpoint
    was trained (routing/dict_num are constructor args, not in the state_dict).
  * --real uses the actual arithmetic-coding round-trip (true bpp); the default
    path uses forward() (estimated bpp). Both report PSNR / MS-SSIM.
  * for balanced_ot, prints mean codebook utilisation (util_frac, col_entropy)
    over the dataset — this feeds the Phase-1 utilisation-vs-N plot.

Example:
  python eval.py --cuda --real --routing balanced_ot --dict-num 512 \
      --checkpoint results/scan_N/balanced_ot_N512/0.013/checkpoint_best.pth.tar \
      --data /path/to/kodak
"""

import argparse
import math
import os
import sys
import time
import warnings

import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms

from models.dcae_ot import DCAE_OT

warnings.filterwarnings("ignore")
torch.set_num_threads(10)


def compute_psnr(a, b):
    return -10 * math.log10(torch.mean((a - b) ** 2).item())


def compute_msssim(a, b):
    return -10 * math.log10(1 - ms_ssim(a, b, data_range=1.0).item())


def compute_bpp(out_net):
    size = out_net["x_hat"].size()
    num_pixels = size[0] * size[2] * size[3]
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


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--real", action="store_true", default=False,
                   help="Use real arithmetic coding (true bpp). Default: forward() estimate.")
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

    img_list = [f for f in os.listdir(args.data) if f[-3:].lower() in ("jpg", "png", "peg")]

    net = DCAE_OT(routing=args.routing, dict_num=args.dict_num,
                  ot_iters=args.ot_iters, ot_eps=args.ot_eps).to(device).eval()

    print("Loading", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    net.load_state_dict(sd)

    PSNR = MSSSIM = Bit_rate = enc_t = dec_t = 0.0
    util_acc = ent_acc = util_count = 0.0
    count = 0

    if args.real:
        net.update(force=True)

    for img_name in img_list:
        img = transforms.ToTensor()(Image.open(os.path.join(args.data, img_name)).convert("RGB")).to(device)
        x = img.unsqueeze(0)
        x_padded, padding = pad(x, p)
        count += 1
        with torch.no_grad():
            if args.real:
                if args.cuda:
                    torch.cuda.synchronize()
                s = time.time()
                out_enc = net.compress(x_padded)
                if args.cuda:
                    torch.cuda.synchronize()
                enc_t += time.time() - s
                s = time.time()
                out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
                if args.cuda:
                    torch.cuda.synchronize()
                dec_t += time.time() - s
                x_hat = crop(out_dec["x_hat"], padding)
                num_pixels = x.size(0) * x.size(2) * x.size(3)
                bpp = sum(len(s_[0]) for s_ in out_enc["strings"]) * 8.0 / num_pixels
            else:
                out_net = net.forward(x_padded)
                out_net["x_hat"].clamp_(0, 1)
                x_hat = crop(out_net["x_hat"], padding)
                bpp = compute_bpp(out_net)
                u = net.routing_utilisation()
                if u is not None:
                    util_acc += u["util_frac"]
                    ent_acc += u["col_entropy"]
                    util_count += 1

            psnr = compute_psnr(x, x_hat)
            msssim = compute_msssim(x, x_hat)
            PSNR += psnr
            MSSSIM += msssim
            Bit_rate += bpp
            print(f"{img_name}: {bpp:.4f} bpp | {psnr:.2f} dB | MS-SSIM {msssim:.2f}")

    print("\n=== Average ===")
    print(f"PSNR     : {PSNR / count:.3f} dB")
    print(f"MS-SSIM  : {MSSSIM / count:.3f}")
    print(f"Bit-rate : {Bit_rate / count:.4f} bpp")
    if args.real:
        print(f"Enc time : {enc_t / count:.3f} s | Dec time : {dec_t / count:.3f} s")
    if util_count > 0:
        print(f"Codebook util_frac : {util_acc / util_count:.3f} "
              f"| col_entropy : {ent_acc / util_count:.3f}  (N={args.dict_num}, {args.routing})")


if __name__ == "__main__":
    main(sys.argv[1:])