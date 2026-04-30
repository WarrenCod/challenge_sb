"""
V-JEPA model: context encoder + EMA target encoder + predictor.

Shapes & conventions
--------------------
Input video:         ``(B, T, C, H, W)``
Tubelet grid:        ``(T', H', W') = (T / tubelet_time, H / tubelet_size, W / tubelet_size)``
Flat token index:    ``i = t * H' * W' + h * W' + w`` (row-major)

One mask (context_ids / target_ids) is shared across the batch each step —
keeps tensor shapes rectangular for speed. Per-sample masking is stronger but
adds padded-gather bookkeeping not worth it on a single-GPU budget.

Loss
----
``smooth_l1( predictor(context_tokens, target_positions),  LN( target_encoder(video)[target_positions] ) )``
Target is layer-normed and detached — both are V-JEPA collapse defenses.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block


class TubeletEmbed(nn.Module):
    """Conv3d patch/tubelet embedding. ``(B, T, C, H, W) -> (B, N, D)``."""

    def __init__(
        self,
        num_frames: int,
        img_size: int,
        tubelet_time: int,
        tubelet_size: int,
        in_chans: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        assert num_frames % tubelet_time == 0, "num_frames must divide by tubelet_time"
        assert img_size % tubelet_size == 0, "img_size must divide by tubelet_size"
        self.t_grid = num_frames // tubelet_time
        self.h_grid = img_size // tubelet_size
        self.w_grid = img_size // tubelet_size
        self.num_tokens = self.t_grid * self.h_grid * self.w_grid
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(tubelet_time, tubelet_size, tubelet_size),
            stride=(tubelet_time, tubelet_size, tubelet_size),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # (B, T, C, H, W) -> (B, C, T, H, W) for Conv3d.
        x = video.permute(0, 2, 1, 3, 4)
        x = self.proj(x)  # (B, D, T', H', W')
        return x.flatten(2).transpose(1, 2)  # (B, N, D)


class VJEPAEncoder(nn.Module):
    """ViT encoder over tubelets, no CLS token."""

    def __init__(
        self,
        num_frames: int,
        img_size: int,
        tubelet_time: int,
        tubelet_size: int,
        in_chans: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.patch_embed = TubeletEmbed(
            num_frames, img_size, tubelet_time, tubelet_size, in_chans, embed_dim
        )
        self.num_tokens = self.patch_embed.num_tokens
        self.t_grid = self.patch_embed.t_grid
        self.h_grid = self.patch_embed.h_grid
        self.w_grid = self.patch_embed.w_grid
        self.embed_dim = embed_dim

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        w = self.patch_embed.proj.weight
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def tokenize(self, video: torch.Tensor) -> torch.Tensor:
        return self.patch_embed(video) + self.pos_embed

    def forward(
        self, video: torch.Tensor, keep_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.tokenize(video)  # (B, N, D)
        if keep_ids is not None:
            x = x[:, keep_ids, :]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class VJEPAPredictor(nn.Module):
    """Narrow transformer that fills in target positions from context tokens."""

    def __init__(
        self,
        num_tokens: int,
        encoder_dim: int,
        pred_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.proj_in = nn.Linear(encoder_dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, pred_dim))
        self.blocks = nn.ModuleList(
            [Block(pred_dim, num_heads, mlp_ratio, qkv_bias=True) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(pred_dim)
        self.proj_out = nn.Linear(pred_dim, encoder_dim)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        context_tokens: torch.Tensor,  # (B, N_ctx, D_enc)
        context_ids: torch.Tensor,     # (N_ctx,)
        target_ids: torch.Tensor,      # (N_tgt,)
    ) -> torch.Tensor:
        B = context_tokens.size(0)
        n_tgt = target_ids.numel()

        ctx = self.proj_in(context_tokens) + self.pos_embed[:, context_ids, :]
        tgt = self.mask_token.expand(B, n_tgt, -1) + self.pos_embed[:, target_ids, :]

        x = torch.cat([ctx, tgt], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        x_tgt = x[:, context_tokens.size(1):, :]
        return self.proj_out(x_tgt)  # (B, N_tgt, D_enc)


class VJEPA(nn.Module):
    """Context encoder + EMA target encoder + predictor."""

    def __init__(
        self,
        num_frames: int = 16,
        img_size: int = 112,
        tubelet_time: int = 2,
        tubelet_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        predictor_embed_dim: int = 192,
        predictor_depth: int = 6,
        predictor_num_heads: int = 6,
    ) -> None:
        super().__init__()
        enc_kwargs = dict(
            num_frames=num_frames,
            img_size=img_size,
            tubelet_time=tubelet_time,
            tubelet_size=tubelet_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        self.context_encoder = VJEPAEncoder(**enc_kwargs)
        self.target_encoder = VJEPAEncoder(**enc_kwargs)
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = VJEPAPredictor(
            num_tokens=self.context_encoder.num_tokens,
            encoder_dim=embed_dim,
            pred_dim=predictor_embed_dim,
            depth=predictor_depth,
            num_heads=predictor_num_heads,
            mlp_ratio=mlp_ratio,
        )

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        for p_ctx, p_tgt in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            p_tgt.data.mul_(momentum).add_(p_ctx.data, alpha=1.0 - momentum)
        for b_ctx, b_tgt in zip(
            self.context_encoder.buffers(), self.target_encoder.buffers()
        ):
            b_tgt.data.copy_(b_ctx.data)

    def forward(
        self,
        video: torch.Tensor,         # (B, T, C, H, W)
        context_ids: torch.Tensor,   # (N_ctx,)
        target_ids: torch.Tensor,    # (N_tgt,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_ctx = self.context_encoder(video, keep_ids=context_ids)
        z_pred = self.predictor(z_ctx, context_ids, target_ids)

        with torch.no_grad():
            z_full = self.target_encoder(video)           # (B, N_total, D)
            z_tgt = z_full[:, target_ids, :]
            z_tgt = F.layer_norm(z_tgt, (z_tgt.size(-1),))
        return z_pred, z_tgt.detach()

    @staticmethod
    def loss(z_pred: torch.Tensor, z_tgt: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(z_pred, z_tgt)

    # --- checkpoint helpers ---

    def context_encoder_state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v for k, v in self.context_encoder.state_dict().items()}

    @torch.no_grad()
    def embedding_std(self, video: torch.Tensor) -> float:
        """Per-dim std of the context encoder output, averaged. Collapse canary."""
        z = self.context_encoder(video)          # (B, N, D)
        z = z.reshape(-1, z.size(-1))
        return float(z.std(dim=0).mean().item())
