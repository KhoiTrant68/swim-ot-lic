"""
adapt_checkpoint.py — load a pretrained DCAE checkpoint into the OT model.

Everything except `dt` is shape-compatible across codebook sizes, so:
  * N == 128 : load the full state_dict (the softmax run should reproduce DCAE).
  * N  > 128 : load everything else; re-initialise `dt` by TILING the pretrained
               128 atoms up to N and adding small noise (warm start >> random).

Usage
-----
    from adapt_checkpoint import load_dcae_into_ot
    net = DCAE_OT(routing=args.routing, dict_num=args.dict_num,
                  ot_iters=args.ot_iters, ot_eps=args.ot_eps)
    load_dcae_into_ot(net, args.checkpoint, dict_num=args.dict_num)
    # then build optimizer FRESH (do not resume DCAE's optimizer for a new N /
    # a new routing op — the moments would not match).
"""

import torch


def load_dcae_into_ot(model, ckpt_path: str, dict_num: int = 128,
                      dt_key: str = "dt", noise_std: float = 0.02,
                      map_location: str = "cpu") -> None:
    ckpt = torch.load(ckpt_path, map_location=map_location)
    sd = ckpt.get("state_dict", ckpt)
    # Strip a possible DDP "module." prefix.
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}

    pre_dt = sd.get(dt_key, None)            # (128, dict_dim) in the released ckpt
    tgt_dt = model.state_dict()[dt_key]      # (dict_num, dict_dim)

    if pre_dt is not None and pre_dt.shape != tgt_dt.shape:
        n_pre, d = pre_dt.shape
        reps = (dict_num + n_pre - 1) // n_pre
        warm = pre_dt.repeat(reps, 1)[:dict_num].clone()
        warm = warm + noise_std * torch.randn_like(warm)   # break tie between clones
        sd[dt_key] = warm
        print(f"[adapt] dt re-init: tiled {n_pre} pretrained atoms -> {dict_num} (+noise {noise_std}).")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    # `missing` should be empty (or only newly-added buffers); `unexpected`
    # should be empty. Print so a silent key mismatch can't hide a 4 dB hole.
    if missing:
        print(f"[adapt] MISSING keys ({len(missing)}): {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"[adapt] UNEXPECTED keys ({len(unexpected)}): {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")
    print(f"[adapt] loaded {ckpt_path} into OT model (N={dict_num}, routing={getattr(model,'routing','?')}).")