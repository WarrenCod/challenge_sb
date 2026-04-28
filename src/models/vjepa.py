"""
V-JEPA Tier-0: frame-pair joint-embedding predictive architecture.

Self-supervised on 4-frame clips. Two ViT-S/16 encoders (context f_theta and
EMA target f_theta_bar) share architecture. A small predictor g_phi takes the
context encoder's output for frame i plus a delta-t embedding, and predicts
the target encoder's patch tokens for frame j.

Loss: smooth_L1 in feature space, scale-aligned via parameter-free LayerNorm
on both prediction and target.

The saved checkpoint stores the EMA target encoder under `encoder_state_dict`
in the same key layout as MAE / iBOT, so Stage 2 can load it via the existing
ViTMAEEncoder without renaming.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block

from models.mae_vit import PatchEmbed, _sincos_2d_posembed


_VIT_VARIANTS = {
    "vit_s_16": dict(embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0, patch_size=16),
    "vit_ti_16": dict(embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0, patch_size=16),
    "vit_b_16": dict(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, patch_size=16),
}


class VJEPAEncoder(nn.Module):
    """ViT-S/16 encoder. Same key layout as ViTMAEEncoder so encoder_state_dict
    is interchangeable. Forward returns ALL tokens (CLS + patches), unlike
    ViTMAEEncoder which returns CLS only — V-JEPA needs patch tokens.
    """

    def __init__(
        self,
        variant: str = "vit_s_16",
        image_size: int = 224,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if variant not in _VIT_VARIANTS:
            raise ValueError(f"Unknown vit variant {variant}. Options: {list(_VIT_VARIANTS)}")
        v = _VIT_VARIANTS[variant]
        self.embed_dim = v["embed_dim"]
        self.patch_size = v["patch_size"]

        self.patch_embed = PatchEmbed(
            img_size=image_size,
            patch_size=v["patch_size"],
            in_chans=3,
            embed_dim=self.embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.register_buffer(
            "pos_embed",
            _sincos_2d_posembed(self.embed_dim, self.patch_embed.grid_size, cls_token=True),
            persistent=False,
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, v["depth"])]
        self.blocks = nn.ModuleList(
            [
                Block(self.embed_dim, v["num_heads"], v["mlp_ratio"], qkv_bias=True, drop_path=dpr[i])
                for i in range(v["depth"])
            ]
        )
        self.norm = nn.LayerNorm(self.embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        """frame: (B, 3, H, W) -> (B, 1+N, D), CLS at index 0."""
        x = self.patch_embed(frame)                          # (B, N, D)
        x = x + self.pos_embed[:, 1:, :]
        cls = self.cls_token + self.pos_embed[:, :1, :]
        cls = cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)                       # (B, 1+N, D)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def encoder_state_dict(self) -> dict:
        """State in the layout consumed by ViTMAEEncoder (Stage 2)."""
        keep_prefixes = ("patch_embed.", "cls_token", "blocks.", "norm.")
        return {k: v for k, v in self.state_dict().items() if k.startswith(keep_prefixes)}


class JEPAPredictor(nn.Module):
    """Narrow self-attention transformer that predicts target patch embeddings.

    Concatenate-and-self-attend pattern (V-JEPA style):
      drop CLS -> proj_in(D_enc -> P) -> +delta_t
      learnable mask queries (mask_token + pos_embed_q + delta_t)
      concat([context, queries]) -> 4x pre-norm Block -> LN(P) -> proj_out -> readout queries
    """

    def __init__(
        self,
        encoder_dim: int = 384,
        predictor_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        num_patches: int = 196,
        n_dt: int = 6,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder_dim = encoder_dim
        self.predictor_dim = predictor_dim
        self.num_patches = num_patches

        self.proj_in = nn.Linear(encoder_dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        grid = int(math.sqrt(num_patches))
        if grid * grid != num_patches:
            raise ValueError(f"num_patches must be a square; got {num_patches}")
        self.register_buffer(
            "pos_embed_q",
            _sincos_2d_posembed(predictor_dim, grid, cls_token=False),
            persistent=False,
        )
        self.dt_embed = nn.Embedding(n_dt, predictor_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(predictor_dim, num_heads, mlp_ratio, qkv_bias=True, drop_path=dpr[i])
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(predictor_dim)              # final LN before proj_out
        self.proj_out = nn.Linear(predictor_dim, encoder_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.dt_embed.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z_i: torch.Tensor, dt_idx: torch.Tensor) -> torch.Tensor:
        """
        z_i:    (B, 1+N, D_enc) full context-encoder output (CLS + patches).
        dt_idx: (B,)            integer in {0..n_dt-1}.

        Returns: (B, N, D_enc).
        """
        B = z_i.shape[0]
        N = self.num_patches
        dt_emb = self.dt_embed(dt_idx)[:, None, :]            # (B, 1, P)

        ctx = self.proj_in(z_i[:, 1:, :])                     # drop CLS -> (B, N, P)
        ctx = ctx + dt_emb

        q = self.mask_token.expand(B, N, -1)                  # (B, N, P)
        q = q + self.pos_embed_q + dt_emb

        x = torch.cat([ctx, q], dim=1)                        # (B, 2N, P)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.proj_out(x)                                  # (B, 2N, D_enc)
        return x[:, N:, :]                                    # query positions only


class VJEPA(nn.Module):
    """Wrapper: context encoder + EMA target encoder + predictor.

    forward(clip) samples per-sample (i, j) pairs with i != j in {0..T-1},
    runs the JEPA loss, and returns dict(loss, cos_sim, target_std).

    EMA update is exposed via ema_step(momentum), called after each optim step.
    """

    def __init__(
        self,
        variant: str = "vit_s_16",
        image_size: int = 224,
        predictor_dim: int = 128,
        predictor_depth: int = 4,
        predictor_heads: int = 4,
        num_frames: int = 4,
        loss_beta: float = 2.0,
    ) -> None:
        super().__init__()
        if num_frames < 2:
            raise ValueError("num_frames must be >= 2 for frame-pair JEPA")
        self.num_frames = num_frames
        self.loss_beta = loss_beta

        self.context_encoder = VJEPAEncoder(variant=variant, image_size=image_size)
        # Target shares architecture; copy weights and freeze.
        self.target_encoder = VJEPAEncoder(variant=variant, image_size=image_size)
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        encoder_dim = self.context_encoder.embed_dim
        num_patches = self.context_encoder.num_patches
        n_dt = 2 * (num_frames - 1)                           # signed dt without 0
        self.predictor = JEPAPredictor(
            encoder_dim=encoder_dim,
            predictor_dim=predictor_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
            num_patches=num_patches,
            n_dt=n_dt,
        )

    def forward(self, clip: torch.Tensor) -> Dict[str, torch.Tensor]:
        """clip: (B, T, 3, H, W). Returns dict(loss, cos_sim, target_std)."""
        B, T = clip.shape[0], clip.shape[1]
        device = clip.device

        # Per-sample (i, j), i != j in {0..T-1}: pick j' in {0..T-2}, shift.
        i = torch.randint(0, T, (B,), device=device)
        j_prime = torch.randint(0, T - 1, (B,), device=device)
        j = j_prime + (j_prime >= i).long()

        arange = torch.arange(B, device=device)
        f_i = clip[arange, i]                                 # (B, 3, H, W)
        f_j = clip[arange, j]

        dt = j - i                                            # in {-(T-1)..-1, +1..+(T-1)}
        dt_idx = (dt + (T - 1)) - (dt > 0).long()             # in {0..2(T-1)-1}

        z_i = self.context_encoder(f_i)                       # (B, 1+N, D), grad
        with torch.no_grad():
            z_j = self.target_encoder(f_j)                    # (B, 1+N, D)
            z_target = z_j[:, 1:, :]                          # drop CLS  (B, N, D)

        z_pred = self.predictor(z_i, dt_idx)                  # (B, N, D)

        # Parameter-free LayerNorm on both sides for scale-invariant matching.
        z_pred_n = F.layer_norm(z_pred, (z_pred.shape[-1],))
        z_target_n = F.layer_norm(z_target, (z_target.shape[-1],))

        loss = F.smooth_l1_loss(z_pred_n, z_target_n.detach(), beta=self.loss_beta)

        with torch.no_grad():
            cos_sim = F.cosine_similarity(z_pred_n, z_target_n, dim=-1).mean()
            target_std = z_target_n.std(dim=0).mean()

        return {"loss": loss, "cos_sim": cos_sim.detach(), "target_std": target_std.detach()}

    @torch.no_grad()
    def ema_step(self, momentum: float) -> None:
        """theta_bar <- m * theta_bar + (1 - m) * theta on the target encoder."""
        for pt, pc in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            pt.data.mul_(momentum).add_(pc.data, alpha=1.0 - momentum)
