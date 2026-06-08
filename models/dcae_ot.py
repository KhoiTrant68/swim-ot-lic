import torch
import torch.nn as nn
from einops import rearrange

from modules.sinkhorn import balanced_sinkhorn
from .dcae import (
    DCAE,
    Scale,
    MultiScaleAggregation,
    ConvolutionalGLU,
)


# ---------------------------------------------------------------------------
# Dictionary cross-attention with routing switch (softmax == exact DCAE)
# ---------------------------------------------------------------------------
class DictCrossAttentionRouted(nn.Module):
    """DCAE's MutiScaleDictionaryCrossAttentionGLU, verbatim except the routing
    step over the N atoms is selectable: 'softmax' (baseline) or 'balanced_ot'."""

    def __init__(self, input_dim, output_dim, mlp_rate=4, head_num=20, qkv_bias=True,
                 routing="softmax", ot_iters=8, ot_eps=1.0):
        super().__init__()
        dict_dim = 32 * head_num
        self.head_num = head_num
        self.routing = routing
        self.ot_iters = ot_iters
        self.ot_eps = ot_eps
        self.last_plan = None  # eval-only cache for utilisation diagnostics

        self.scale = nn.Parameter(torch.ones(head_num, 1, 1))
        self.x_trans = nn.Linear(input_dim, dict_dim, bias=qkv_bias)
        self.ln_scale = nn.LayerNorm(dict_dim)
        self.msa = MultiScaleAggregation(dict_dim)
        self.lnx = nn.LayerNorm(dict_dim)
        self.q_trans = nn.Linear(dict_dim, dict_dim, bias=qkv_bias)
        self.dict_ln = nn.LayerNorm(dict_dim)
        self.k = nn.Linear(dict_dim, dict_dim, bias=qkv_bias)
        self.linear = nn.Linear(dict_dim, dict_dim, bias=qkv_bias)
        self.ln_mlp = nn.LayerNorm(dict_dim)
        self.mlp = ConvolutionalGLU(dict_dim, mlp_rate * dict_dim)
        self.output_trans = nn.Sequential(nn.Linear(dict_dim, output_dim))
        self.softmax = nn.Softmax(dim=-1)

        self.res_scale_1 = Scale(dict_dim, init_value=1.0)
        self.res_scale_2 = Scale(dict_dim, init_value=1.0)
        self.res_scale_3 = Scale(dict_dim, init_value=1.0)

    def forward(self, x, dt):
        B, C, H, W = x.size()
        x = rearrange(x, "b c h w -> b h w c")
        x = self.x_trans(x)
        x = self.msa(self.ln_scale(x)) + self.res_scale_1(x)

        shortcut = x
        x = self.lnx(x)
        x = self.q_trans(x)
        x = rearrange(x, "b h w c -> b c h w")

        q = rearrange(x, "b (e c) h w -> b e (h w) c", e=self.head_num)
        dt = self.dict_ln(dt)
        k = self.k(dt)
        k = rearrange(k, "b n (e c) -> b e n c", e=self.head_num)
        dt = rearrange(dt, "b n (e c) -> b e n c", e=self.head_num)  # values
        self.scale = self.scale.to(q.device)

        sim = torch.einsum("benc,bedc->bend", q, k)   # (B, heads, HW, N)
        gain = sim * self.scale

        # ---- the ONLY change vs DCAE: routing over the N atoms ----
        if self.routing == "softmax":
            probs = self.softmax(gain)
        elif self.routing == "balanced_ot":
            P = balanced_sinkhorn(gain, n_iters=self.ot_iters, eps=self.ot_eps)
            probs = P / (P.sum(dim=-1, keepdim=True) + 1e-9)  # row-normalise
            if not self.training:
                self.last_plan = P.detach()
        else:
            raise ValueError(f"unknown routing: {self.routing}")
        # -----------------------------------------------------------

        output = torch.einsum("bend,bedc->benc", probs, dt)
        output = rearrange(output, "b e (h w) c -> b h w (e c) ", h=H, w=W)
        output = self.linear(output) + self.res_scale_2(shortcut)
        output = self.mlp(self.ln_mlp(output)) + self.res_scale_3(output)
        output = self.output_trans(output)
        output = rearrange(output, "b h w c -> b c h w")
        return output


# ---------------------------------------------------------------------------
# DCAE_OT: DCAE + selectable routing + selectable codebook size
# ---------------------------------------------------------------------------
class DCAE_OT(DCAE):
    def __init__(self, routing="softmax", dict_num=128, dict_head_num=20,
                 ot_iters=8, ot_eps=1.0, **kwargs):
        # Build the full DCAE first (uses default 128-atom softmax dict)...
        super().__init__(**kwargs)
        # ...then replace ONLY the dictionary pieces.
        self.routing = routing
        self.dict_num = dict_num
        dict_dim = 32 * dict_head_num
        M = self.M

        self.dt = nn.Parameter(torch.randn([dict_num, dict_dim]))
        self.dt_cross_attention = nn.ModuleList(
            DictCrossAttentionRouted(
                input_dim=M * 2 + (M // self.num_slices) * i,
                output_dim=M,
                head_num=dict_head_num,
                routing=routing,
                ot_iters=ot_iters,
                ot_eps=ot_eps,
            )
            for i in range(self.num_slices)
        )

    @torch.no_grad()
    def routing_utilisation(self, thresh_ratio: float = 0.01):
        """Aggregate plan_utilisation over slices from the LAST eval forward.
        Returns None for softmax routing (no transport plan cached)."""
        from modules.sinkhorn import plan_utilisation
        plans = [a.last_plan for a in self.dt_cross_attention if a.last_plan is not None]
        if not plans:
            return None
        stats = [plan_utilisation(p, thresh_ratio) for p in plans]
        util = sum(s["util_frac"] for s in stats) / len(stats)
        ent = sum(s["col_entropy"] for s in stats) / len(stats)
        return {"util_frac": util, "col_entropy": ent, "num_slices": len(stats)}