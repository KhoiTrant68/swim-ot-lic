"""
sinkhorn.py — Balanced entropic-OT routing (log-domain Sinkhorn-Knopp).

Given a gain/similarity tensor between M spatial tokens (rows) and N dictionary
atoms (cols), it returns a transport plan whose COLUMN marginal is forced
uniform — every atom receives equal total mass. That load-balancing is the
one property plain softmax cannot guarantee, and it is what keeps a large
codebook from collapsing into a few dominant atoms.

Anti-collapse-via-balanced-assignment precedent: Cuturi (2013) entropic OT;
Asano et al. SeLa (2020); Caron et al. SwAV (2020).
"""

import math
import torch


def balanced_sinkhorn(gain: torch.Tensor, n_iters: int = 8, eps: float = 1.0) -> torch.Tensor:
    """
    Parameters
    ----------
    gain : (..., M, N)  similarity logits (already multiplied by the learnable
                        temperature `scale`). Higher = token/atom more compatible.
    n_iters : Sinkhorn iterations (fully differentiable; 6-10 is plenty here).
    eps : entropic temperature. With eps=1.0 the learnable `scale` is the sole
          sharpness knob (recommended). Lower eps = sharper transport.

    Returns
    -------
    P : (..., M, N) entropic-OT plan with
          row marginal  P.sum(-1) ~ 1/M   (each token equal demand)
          col marginal  P.sum(-2) ~ 1/N   (each atom equal supply  <- anti-collapse)
    Caller should row-normalise before aggregating values so each token gets a
    convex combination:  probs = P / P.sum(-1, keepdim=True).

    Solves   max_P <P, gain> + eps*H(P)   s.t. the uniform marginals above,
    in the log domain (stable for sharp / small-eps routing).
    """
    *batch, M, N = gain.shape
    log_K = gain / eps
    neg_log_M = -math.log(M)
    neg_log_N = -math.log(N)
    log_u = gain.new_zeros((*batch, M))
    log_v = gain.new_zeros((*batch, N))
    for _ in range(n_iters):
        log_u = neg_log_M - torch.logsumexp(log_K + log_v.unsqueeze(-2), dim=-1)
        log_v = neg_log_N - torch.logsumexp(log_K + log_u.unsqueeze(-1), dim=-2)
    return torch.exp(log_K + log_u.unsqueeze(-1) + log_v.unsqueeze(-2))


@torch.no_grad()
def plan_utilisation(P: torch.Tensor, thresh_ratio: float = 0.01) -> dict:
    """
    Diagnostics for the make-or-break plot. P: (..., M, N).

    util_frac   : fraction of atoms whose mean column mass exceeds
                  thresh_ratio * (1/N)  -> roughly the "alive" atoms.
    col_entropy : column-marginal entropy normalised to [0, 1]
                  (1 = perfectly uniform usage; low = a few atoms dominate).

    For the scaling sweep, plot BOTH vs N for softmax and balanced_ot:
    softmax is expected to drop util_frac as N grows; OT should hold it ~1.
    """
    N = P.shape[-1]
    flat = P.reshape(-1, N)
    col_mass = flat.mean(0)
    col_mass = col_mass / (col_mass.sum() + 1e-12)
    util_frac = (col_mass > thresh_ratio / N).float().mean().item()
    ent = -(col_mass * (col_mass + 1e-12).log()).sum().item()
    return {"util_frac": util_frac, "col_entropy": ent / math.log(N)}