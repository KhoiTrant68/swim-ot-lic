import re

import torch


# CompressAI EntropyBottleneck attribute names changed in newer versions:
# old: _bias{i} / _factor{i} / _matrix{i}  (flat Parameter/Buffer per index)
# new: biases.{i} / factors.{i} / matrices.{i}  (ParameterList)
# Pretrained DCAE checkpoints use the old layout; current compressai expects
# the new one. Without this rename, 14 entropy_bottleneck keys silently stay
# at random init, which destroys bpp.
_LEGACY_RENAME = {"_bias": "biases", "_factor": "factors", "_matrix": "matrices"}
_LEGACY_PAT = re.compile(r"^(.*?)(_bias|_factor|_matrix)(\d+)$")


def _rename_legacy_compressai_keys(sd: dict) -> dict:
    out, renamed = {}, 0
    for k, v in sd.items():
        m = _LEGACY_PAT.match(k)
        if m:
            prefix, attr, idx = m.groups()
            out[f"{prefix}{_LEGACY_RENAME[attr]}.{idx}"] = v
            renamed += 1
        else:
            out[k] = v
    if renamed:
        print(f"[adapt] renamed {renamed} legacy compressai keys "
              f"(_bias/_factor/_matrix{{i}} -> biases/factors/matrices.{{i}}).")
    return out


def load_dcae_into_ot(model, ckpt_path: str, dict_num: int = 128,
                      dt_key: str = "dt", noise_std: float = 0.02,
                      map_location: str = "cpu") -> None:
    ckpt = torch.load(ckpt_path, map_location=map_location)
    sd = ckpt.get("state_dict", ckpt)
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    sd = _rename_legacy_compressai_keys(sd)

    # --- re-init dt if the codebook size differs from the checkpoint ---
    pre_dt = sd.get(dt_key, None)
    tgt_dt = model.state_dict()[dt_key]
    if pre_dt is not None and pre_dt.shape != tgt_dt.shape:
        n_pre, d = pre_dt.shape
        reps = (dict_num + n_pre - 1) // n_pre
        warm = pre_dt.repeat(reps, 1)[:dict_num].clone()
        warm = warm + noise_std * torch.randn_like(warm)
        sd[dt_key] = warm
        print(f"[adapt] dt re-init: tiled {n_pre} pretrained atoms -> {dict_num} (+noise {noise_std}).")

    # --- report key diff ourselves (DCAE.load_state_dict returns None) ---
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(sd.keys())
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    if missing:
        print(f"[adapt] MISSING keys ({len(missing)}): {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"[adapt] UNEXPECTED keys ({len(unexpected)}): {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")

    # --- load WITHOUT unpacking; DCAE's override resizes the CDF buffers ---
    try:
        model.load_state_dict(sd, strict=False)
    except TypeError:
        # Some override versions don't accept strict= ; retry without it.
        model.load_state_dict(sd)

    print(f"[adapt] loaded {ckpt_path} into OT model "
          f"(N={dict_num}, routing={getattr(model, 'routing', '?')}).")