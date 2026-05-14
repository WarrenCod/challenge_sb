"""Divided space-time block wrapper for the VideoMAE flat-layout encoder.

Adds a temporal-attention pre-pass to a timm ``Block`` from the VideoMAE
backbone. The encoder operates on flat ``(B, N=T'*H'*W', D)`` tokens; this
wrapper reshapes to ``(B, T', H'·W', D)``, adds a learnable temporal-pos
embed, permutes to ``(B*H'·W', T', D)`` for a per-patch temporal MHA, and
flattens back before delegating to the wrapped spatial block.

Step-0 invariant: the temporal MHA's output projection AND the temporal-pos
parameter are both zero-initialised, so at step 0 the wrapped block's
forward is bit-identical to the underlying spatial Block's. Any drift from
the meanpool baseline must be earned by gradient updates.

VideoMAE caveat vs Jabiru's exp2m: Jabiru applied this to a per-frame ViT
where the temporal axis is T=4 (one position per frame). VideoMAE with
tubelet_time=2 yields T'=2 (each tubelet already fuses 2 frames in the
patch embed). The temporal MHA here therefore sees only 2 positions per
patch — a thinner lever than Jabiru's 4. Expect a smaller gain (+0.3..1.0
pp vs Jabiru's +5.27 pp).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VideoMAESpaceTimeBlock(nn.Module):
    """Wraps a timm Block with a temporal-attention pre-pass over T'."""

    def __init__(
        self,
        spatial_block: nn.Module,
        embed_dim: int,
        num_heads: int,
        t_grid: int,
        hw_count: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.t_grid = int(t_grid)
        self.hw_count = int(hw_count)
        self.norm_t = nn.LayerNorm(embed_dim)
        self.attn_t = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        nn.init.zeros_(self.attn_t.out_proj.weight)
        nn.init.zeros_(self.attn_t.out_proj.bias)
        # (1, T', 1, D) so it broadcasts over the spatial patch axis after reshape.
        self.temporal_pos = nn.Parameter(torch.zeros(1, self.t_grid, 1, embed_dim))
        self.block = spatial_block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N=T'*H'*W', D)
        B, N, D = x.shape
        if N != self.t_grid * self.hw_count:
            raise RuntimeError(
                f"VideoMAESpaceTimeBlock: N={N} ≠ T'·H'·W' = {self.t_grid}·{self.hw_count}"
            )
        x_4d = x.view(B, self.t_grid, self.hw_count, D) + self.temporal_pos
        # (B*H'·W', T', D) for per-patch temporal attention
        x_perm = x_4d.permute(0, 2, 1, 3).reshape(B * self.hw_count, self.t_grid, D)
        h = self.norm_t(x_perm)
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            h32 = h.float()
            attn_out, _ = self.attn_t(h32, h32, h32, need_weights=False)
        x_perm = x_perm + attn_out.to(x_perm.dtype)
        # Back to flat (B, N, D)
        x_flat = (
            x_perm.view(B, self.hw_count, self.t_grid, D)
            .permute(0, 2, 1, 3)
            .reshape(B, N, D)
        )
        return self.block(x_flat)
