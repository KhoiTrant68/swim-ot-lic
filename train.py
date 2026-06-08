import os
os.environ.setdefault("TMPDIR", "/tmp")
import argparse
import math
import random
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from compressai.datasets import ImageFolder
from pytorch_msssim import ms_ssim
from torch.utils.tensorboard import SummaryWriter

from models.dcae_ot import DCAE_OT
from modules.adapt_checkpoint import load_dcae_into_ot

torch.set_num_threads(8)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.0)


class RateDistortionLoss(nn.Module):
    """DCAE's loss, unchanged: lambda*255^2*MSE + bpp  (or ms-ssim variant)."""

    def __init__(self, lmbda=1e-2, type="mse"):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.type = type

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        out["bpp_loss"] = sum(
            (torch.log(lk).sum() / (-math.log(2) * num_pixels))
            for lk in output["likelihoods"].values()
        )
        if self.type == "mse":
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        else:
            out["ms_ssim_loss"] = compute_msssim(output["x_hat"], target)
            out["loss"] = self.lmbda * (1 - out["ms_ssim_loss"]) + out["bpp_loss"]
        return out


class AverageMeter:
    def __init__(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def configure_optimizers(net, args):
    params = {n for n, p in net.named_parameters()
              if not n.endswith(".quantiles") and p.requires_grad}
    aux_params = {n for n, p in net.named_parameters()
                  if n.endswith(".quantiles") and p.requires_grad}
    d = dict(net.named_parameters())
    assert len(params & aux_params) == 0
    assert len(params | aux_params) - len(d.keys()) == 0
    optimizer = optim.Adam((d[n] for n in sorted(params)), lr=args.learning_rate)
    aux_optimizer = optim.Adam((d[n] for n in sorted(aux_params)), lr=args.aux_learning_rate)
    return optimizer, aux_optimizer


def train_one_epoch(model, criterion, loader, optimizer, aux_optimizer, epoch,
                    clip_max_norm, train_sampler, type, lr_scheduler):
    model.train()
    device = next(model.parameters()).device
    if torch.cuda.device_count() > 1:
        train_sampler.set_epoch(epoch)
    for i, d in enumerate(loader):
        d = d.to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        out_net = model(d)
        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        aux = model.module.aux_loss() if torch.cuda.device_count() > 1 else model.aux_loss()
        aux.backward()
        aux_optimizer.step()
        if (i + 1) % 100 == 0:
            msg = (f"Epoch {epoch} [{(i+1)*len(d)}/{len(loader.dataset)}] "
                   f"loss {out_criterion['loss'].item():.3f} "
                   f"bpp {out_criterion['bpp_loss'].item():.3f} "
                   f"aux {aux.item():.2f} lr {lr_scheduler.get_last_lr()[0]:.2e}")
            if type == "mse":
                msg += f" mse {out_criterion['mse_loss'].item():.4f}"
            print(msg, flush=True)


@torch.no_grad()
def test_epoch(epoch, loader, model, criterion, type):
    model.eval()
    device = next(model.parameters()).device
    loss, bpp, dist_m, aux_m = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    for d in loader:
        d = d.to(device)
        out_net = model(d)
        out_criterion = criterion(out_net, d)
        loss.update(out_criterion["loss"])
        bpp.update(out_criterion["bpp_loss"])
        aux_m.update(model.module.aux_loss() if torch.cuda.device_count() > 1 else model.aux_loss())
        dist_m.update(out_criterion["mse_loss"] if type == "mse" else (1 - out_criterion["ms_ssim_loss"]))
    psnr = -10 * math.log10(dist_m.avg) if type == "mse" and dist_m.avg > 0 else float("nan")
    print(f"[Test {epoch}] loss {loss.avg:.4f} | bpp {bpp.avg:.4f} | "
          f"dist {dist_m.avg:.5f} | est-PSNR {psnr:.2f} dB | aux {aux_m.avg:.2f}", flush=True)
    return loss.avg


def save_checkpoint(state, is_best, epoch, save_path):
    torch.save(state, save_path + "checkpoint_latest.pth.tar")
    if epoch % 5 == 0:
        torch.save(state, save_path + f"{epoch}_checkpoint.pth.tar")
    if is_best:
        torch.save(state, save_path + "checkpoint_best.pth.tar")


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--local-rank", default=int(os.getenv("LOCAL_RANK", -1)), type=int)
    p.add_argument("-d", "--dataset", type=str, required=True)
    p.add_argument("-e", "--epochs", default=40, type=int)
    p.add_argument("-lr", "--learning-rate", default=1e-4, type=float)
    p.add_argument("-n", "--num-workers", default=16, type=int)
    p.add_argument("--lambda", dest="lmbda", type=float, default=0.013)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--test-batch-size", type=int, default=8)
    p.add_argument("--aux-learning-rate", default=1e-3, type=float)
    p.add_argument("--patch-size", type=int, nargs=2, default=(256, 256))
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--save", action="store_true", default=True)
    p.add_argument("--seed", type=float, default=100)
    p.add_argument("--clip_max_norm", default=1.0, type=float)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--type", type=str, default="mse", choices=["mse", "ms-ssim"])
    p.add_argument("--save_path", type=str, required=True)
    p.add_argument("--lr_epoch", nargs="+", type=int, default=[30])
    # ── OT-routing experiment args ───────────────────────────────────────
    p.add_argument("--routing", type=str, default="softmax",
                   choices=["softmax", "balanced_ot"])
    p.add_argument("--dict-num", dest="dict_num", type=int, default=128)
    p.add_argument("--ot-iters", dest="ot_iters", type=int, default=8)
    p.add_argument("--ot-eps", dest="ot_eps", type=float, default=1.0)
    p.add_argument("--finetune", action="store_true", default=False,
                   help="Load DCAE weights only (re-init dt if N!=128); fresh "
                        "optimizer/scheduler/epoch. Use this for the sweep.")
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    for a in vars(args):
        print(a, ":", getattr(args, a))
    type = args.type
    save_path = os.path.join(args.save_path, str(args.lmbda)) + "/"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path + "tensorboard/", exist_ok=True)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    writer = SummaryWriter(save_path + "tensorboard/")

    train_tf = transforms.Compose([transforms.RandomCrop(args.patch_size), transforms.ToTensor()])
    test_tf = transforms.Compose([transforms.CenterCrop(args.patch_size), transforms.ToTensor()])
    train_dataset = ImageFolder(args.dataset, split="train", transform=train_tf)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_tf)

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    else:
        device = "cuda" if args.cuda else "cpu"

    # Build model, then load weights BEFORE the DDP wrap (avoids "module." prefix).
    net = DCAE_OT(routing=args.routing, dict_num=args.dict_num,
                  ot_iters=args.ot_iters, ot_eps=args.ot_eps)
    last_epoch = 0
    resume_opt = False
    if args.checkpoint:
        if args.finetune:
            load_dcae_into_ot(net, args.checkpoint, dict_num=args.dict_num)
        else:
            ckpt = torch.load(args.checkpoint, map_location="cpu")
            sd = ckpt.get("state_dict", ckpt)
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            net.load_state_dict(sd)
            last_epoch = ckpt.get("epoch", -1) + 1
            resume_opt = True
    net = net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = nn.parallel.DistributedDataParallel(
            net, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=True)
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset)

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler if torch.cuda.device_count() > 1 else None,
        shuffle=(torch.cuda.device_count() == 1),
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(
        test_dataset,
        sampler=test_sampler if torch.cuda.device_count() > 1 else None,
        batch_size=args.test_batch_size, num_workers=args.num_workers,
        shuffle=False, pin_memory=True)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.lr_epoch, gamma=0.1, last_epoch=-1)
    criterion = RateDistortionLoss(lmbda=args.lmbda, type=type).to(device)

    if resume_opt:
        ckpt = torch.load(args.checkpoint, map_location=device)
        optimizer.load_state_dict(ckpt["optimizer"])
        aux_optimizer.load_state_dict(ckpt["aux_optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])

    best_loss = float("inf")
    train_sampler_ref = train_sampler if (args.cuda and torch.cuda.device_count() > 1) else None
    for epoch in range(last_epoch, args.epochs):
        train_one_epoch(net, criterion, train_loader, optimizer, aux_optimizer,
                        epoch, args.clip_max_norm, train_sampler_ref, type, lr_scheduler)
        loss = test_epoch(epoch, test_loader, net, criterion, type)
        writer.add_scalar("test_loss", loss, epoch)
        lr_scheduler.step()
        global_rank = dist.get_rank() if (args.cuda and torch.cuda.device_count() > 1) else 0
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        if args.save and global_rank == 0:
            save_checkpoint(
                {"epoch": epoch,
                 "state_dict": (net.module if hasattr(net, "module") else net).state_dict(),
                 "loss": loss, "optimizer": optimizer.state_dict(),
                 "aux_optimizer": aux_optimizer.state_dict(),
                 "lr_scheduler": lr_scheduler.state_dict(),
                 "routing": args.routing, "dict_num": args.dict_num,
                 "ot_iters": args.ot_iters, "ot_eps": args.ot_eps, "lambda": args.lmbda},
                is_best, epoch, save_path)


if __name__ == "__main__":
    main(sys.argv[1:])