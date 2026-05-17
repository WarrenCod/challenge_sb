"""
V-JEPA: spacetime masked-feature prediction. Three recipes share one module:

  * "block3d" + "smooth_l1_lnboth"  — legacy (Tier-1/Tier-2/v3a). Single
    random 3D-block mask per clip, smooth_L1 with parameter-free LayerNorm
    on both prediction and target.

  * "vjepa" + "l1_lntarget"         — V-JEPA 1 port (v4). M distinct masks
    per clip per step (multi-mask), each the union of `n_long_blocks`
    long-range tubes (large spatial scale) and `n_short_blocks` short-range
    tubes (small spatial scale), all lifted across the full T_tok dimension.
    Loss is L1 with parameter-free LayerNorm on the target only — matches
    the V-JEPA 1 paper modulo flip (disabled here for SSv2 direction).

  * "tube_block_v7" + "jepa_pixel" — v7 hybrid. One large rectangular tube
    masked across all T_tok slices at ~90% ratio (motion-forcing), and a
    dual-objective predictor: feature-prediction (L1 vs LN target) PLUS
    pixel reconstruction (L2 vs per-patch-LN RGB). Pixel target is
    non-gameable, anchoring against the appearance-shortcut collapse that
    capped v3a→v6 probes at ~9.6%.

Multi-mask (M ≥ 2) reuses the encoder forward across M predictor passes:
the encoder is the dominant cost, so each step yields ~M× learning signal
for ~1.3× wall-clock. This is the central sample-efficiency lever from
V-JEPA 1 / V-JEPA 2.

Encoder layout (`VJEPAEncoder`, `TubeletPatchEmbed`, state-dict key prefix
list) is unchanged across recipes. The saved EMA-encoder dump still uses
the MAE/iBOT key layout (`encoder_state_dict`), so Stage 2 loads it via the
existing `ViTMAEEncoder` without renaming.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, Mlp
from timm.models.vision_transformer import Block

from models._vit_utils import PatchEmbed, _sincos_2d_posembed, _sincos_3d_posembed


_VIT_VARIANTS = {
    "vit_s_16": dict(embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0, patch_size=16),
    "vit_ti_16": dict(embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0, patch_size=16),
    "vit_b_16": dict(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, patch_size=16),
}


class TubeletPatchEmbed(nn.Module):
    """3D patch embedding: Conv3d kernel/stride = (tubelet_size, patch, patch).

    Input:  (B, C, T, H, W). Output: (B, T_tok * H_p * W_p, D), with
    T_tok = T // tubelet_size and H_p = W_p = img_size // patch_size.
    Mirrors the layout used by VideoMAE / V-JEPA-2 / TubeViT.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 384,
    ) -> None:
        super().__init__()
        if tubelet_size < 1:
            raise ValueError(f"tubelet_size must be >= 1; got {tubelet_size}")
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.grid_size = img_size // patch_size
        self.num_spatial_patches = self.grid_size ** 2
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        # clip: (B, C, T, H, W) -> (B, D, T_tok, H_p, W_p) -> (B, T_tok*H_p*W_p, D)
        x = self.proj(clip)
        return x.flatten(2).transpose(1, 2)


class VJEPAEncoder(nn.Module):
    """ViT encoder used by V-JEPA. Two modes:

    * `tubelet_size == 1` (Tier-1): standard 2D per-frame ViT. Forward takes
      a single frame `(B, 3, H, W)` and returns `(B, 1+N, D)`. Frames are
      processed independently by the caller (fold T into batch).
    * `tubelet_size > 1` (Tier-2): 3D-tubelet patch embed + 3D pos embed +
      joint space-time attention. Forward takes the whole clip
      `(B, T, 3, H, W)` and returns `(B, 1 + T_tok*N_sp, D)`, with
      `T_tok = T // tubelet_size`.

    State-dict layout is preserved across modes (`patch_embed.proj.weight`,
    `cls_token`, `blocks.*`, `norm.*`), so an encoder dump is loadable by
    `ViTMAEEncoder` if the latter is constructed with the same
    `tubelet_size` / `num_frames`.
    """

    def __init__(
        self,
        variant: str = "vit_s_16",
        image_size: int = 224,
        drop_path_rate: float = 0.0,
        tubelet_size: int = 1,
        num_frames: int = 4,
    ) -> None:
        super().__init__()
        if variant not in _VIT_VARIANTS:
            raise ValueError(f"Unknown vit variant {variant}. Options: {list(_VIT_VARIANTS)}")
        if tubelet_size < 1:
            raise ValueError(f"tubelet_size must be >= 1; got {tubelet_size}")
        if tubelet_size > 1 and num_frames % tubelet_size != 0:
            raise ValueError(f"num_frames {num_frames} must be divisible by tubelet_size {tubelet_size}")
        v = _VIT_VARIANTS[variant]
        self.embed_dim = v["embed_dim"]
        self.patch_size = v["patch_size"]
        self.tubelet_size = tubelet_size
        self.num_frames = num_frames
        self.t_tokens = num_frames // tubelet_size if tubelet_size > 1 else 1

        if tubelet_size == 1:
            self.patch_embed = PatchEmbed(
                img_size=image_size,
                patch_size=v["patch_size"],
                in_chans=3,
                embed_dim=self.embed_dim,
            )
            self.num_spatial_patches = self.patch_embed.num_patches
            self.grid_size = self.patch_embed.grid_size
            pos = _sincos_2d_posembed(self.embed_dim, self.grid_size, cls_token=True)
        else:
            self.patch_embed = TubeletPatchEmbed(
                img_size=image_size,
                patch_size=v["patch_size"],
                tubelet_size=tubelet_size,
                in_chans=3,
                embed_dim=self.embed_dim,
            )
            self.num_spatial_patches = self.patch_embed.num_spatial_patches
            self.grid_size = self.patch_embed.grid_size
            pos = _sincos_3d_posembed(self.embed_dim, self.t_tokens, self.grid_size, cls_token=True)

        # `num_patches` keeps Tier-1 semantics (spatial patch count) so the
        # mask sampler / predictor can read it without branching.
        self.num_patches = self.num_spatial_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.register_buffer("pos_embed", pos, persistent=False)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tier-1 (tubelet=1): `frame` (B, 3, H, W) -> (B, 1+N, D).
        Tier-2 (tubelet>1):   `clip`  (B, T, 3, H, W) -> (B, 1+T_tok*N, D)."""
        if self.tubelet_size == 1:
            x = self.patch_embed(x)                          # (B, N, D)
        else:
            # (B, T, C, H, W) -> (B, C, T, H, W) for Conv3d
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            x = self.patch_embed(x)                          # (B, T_tok*N, D)
        x = x + self.pos_embed[:, 1:, :]
        cls = self.cls_token + self.pos_embed[:, :1, :]
        cls = cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)                       # (B, 1+(T_tok*)N, D)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def encoder_state_dict(self) -> dict:
        """State in the layout consumed by ViTMAEEncoder (Stage 2)."""
        keep_prefixes = ("patch_embed.", "cls_token", "blocks.", "norm.")
        return {k: v for k, v in self.state_dict().items() if k.startswith(keep_prefixes)}


class JEPAPredictor(nn.Module):
    """Spacetime predictor over (B, T*N, D_enc) context tokens with a binary mask.

    Replaces masked positions with a learnable mask_token + (space, time) pos
    embed, attends jointly across the full T*N sequence, and projects masked
    positions back to encoder dim. Visible-context positions also evolve through
    the predictor blocks (vs the original V-JEPA where they pass through frozen);
    this is a small simplification that costs little and avoids gather/scatter.
    """

    def __init__(
        self,
        encoder_dim: int = 384,
        predictor_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_patches: int = 196,
        max_frames: int = 8,
        drop_path_rate: float = 0.0,
        pixel_out_dim: int = 0,
    ) -> None:
        super().__init__()
        self.encoder_dim = encoder_dim
        self.predictor_dim = predictor_dim
        self.num_patches = num_patches
        self.max_frames = max_frames
        self.pixel_out_dim = pixel_out_dim

        self.proj_in = nn.Linear(encoder_dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))

        grid = int(math.sqrt(num_patches))
        if grid * grid != num_patches:
            raise ValueError(f"num_patches must be a square; got {num_patches}")
        self.register_buffer(
            "pos_embed_space",
            _sincos_2d_posembed(predictor_dim, grid, cls_token=False).unsqueeze(1),  # (1, 1, N, P)
            persistent=False,
        )
        # Learned time pos-embed, zero-init: (1, max_frames, 1, P).
        self.pos_embed_time = nn.Parameter(torch.zeros(1, max_frames, 1, predictor_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(predictor_dim, num_heads, mlp_ratio, qkv_bias=True, drop_path=dpr[i])
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(predictor_dim)
        self.proj_out = nn.Linear(predictor_dim, encoder_dim)

        # Optional pixel-reconstruction head (v7). Narrow projection from the
        # predictor's normalized output to per-tubelet RGB. Kept lightweight
        # (LN + Linear) so the encoder is forced to do the semantic work; a
        # deep decoder would let the predictor solve pixel-recon on its own.
        if pixel_out_dim > 0:
            self.pixel_proj = nn.Linear(predictor_dim, pixel_out_dim)
        else:
            self.pixel_proj = None

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_time, std=0.02)
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
        z_ctx: torch.Tensor,
        masks: torch.Tensor,
        T: int,
    ):
        """
        z_ctx: (B, T*N, D_enc) — context-encoder patch tokens, all positions.
        masks: (B, T*N)  bool  — True at positions to predict (constant count per row).
        T:     int             — number of frames.

        Returns: (B, n_masked, D_enc) — feature predictions at the masked
        positions only. If pixel_proj is enabled, returns a tuple
        ``(z_feat, z_pix)`` where ``z_pix`` has shape (B, n_masked, pixel_out_dim).
        """
        B, S, D = z_ctx.shape
        N = self.num_patches
        if S != T * N:
            raise ValueError(f"z_ctx has {S} tokens; expected T*N={T * N}")
        if T > self.max_frames:
            raise ValueError(f"T={T} exceeds max_frames={self.max_frames}")

        z = self.proj_in(z_ctx).reshape(B, T, N, self.predictor_dim)
        pos = self.pos_embed_space + self.pos_embed_time[:, :T]              # (1, T, N, P)
        z = z + pos

        # Replace masked positions with mask_token + same pos embed.
        mask_q = (self.mask_token.unsqueeze(1) + pos).expand(B, -1, -1, -1)  # (B, T, N, P)
        masks_4d = masks.reshape(B, T, N, 1)
        z = torch.where(masks_4d, mask_q, z)
        z = z.reshape(B, T * N, self.predictor_dim)

        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)                                                      # (B, T*N, P)
        z_feat = self.proj_out(z)                                             # (B, T*N, D_enc)

        # Gather predictions at masked positions only. masks has fixed count
        # per row, so the reshape is well-defined.
        n_masked = int(masks[0].sum().item())
        z_feat_at_mask = z_feat[masks].reshape(B, n_masked, D)

        if self.pixel_proj is None:
            return z_feat_at_mask
        z_pix = self.pixel_proj(z)                                            # (B, T*N, pixel_out_dim)
        z_pix_at_mask = z_pix[masks].reshape(B, n_masked, self.pixel_out_dim)
        return z_feat_at_mask, z_pix_at_mask


class _CrossAttnBlock(nn.Module):
    """Pre-norm cross-attention block: queries cross-attend to a keys/values
    sequence, then a per-query MLP. Mirrors timm's Block layout but with
    asymmetric attention (different sequences for Q vs K=V).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=proj_drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q_n = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        attn_out, _ = self.attn(q_n, kv_n, kv_n, need_weights=False)
        q = q + self.drop_path1(self.proj_drop(attn_out))
        q = q + self.drop_path2(self.mlp(self.norm2(q)))
        return q


class HybridCrossSelfPredictor(nn.Module):
    """V-JEPA 2-style predictor with explicit extract-then-refine structure.

    Phase A — extraction (``n_cross`` cross-attention blocks):
        Mask-token queries (one per masked position, with space/time pos-embed)
        cross-attend to the visible context tokens. The encoder is forced to
        produce visible features rich enough to "explain" the masked positions;
        masked queries cannot exchange information among themselves in this
        phase, which removes the inline-self-attention shortcut that V-JEPA 1
        predictors are known to take.

    Phase B — refinement (``n_self`` self-attention blocks):
        The joined sequence [visible_tokens ; refined_queries] runs through
        timm self-attention blocks for global consistency. Predictions are read
        from the trailing query positions, projected back to encoder dim, and
        optionally also projected to per-tubelet RGB for the v7 pixel anchor.

    Same external interface as ``JEPAPredictor.forward``: takes
    ``(z_ctx, masks, T)`` and returns ``z_feat_at_mask`` (or a tuple with
    ``z_pix_at_mask`` if ``pixel_out_dim > 0``). All samples in the batch are
    assumed to have the same mask count (``self.n_masked``), which is
    guaranteed by the trim/pad in every mask sampler.
    """

    def __init__(
        self,
        encoder_dim: int = 384,
        predictor_dim: int = 384,
        n_cross: int = 3,
        n_self: int = 2,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        num_patches: int = 196,
        max_frames: int = 8,
        drop_path_rate: float = 0.0,
        pixel_out_dim: int = 0,
    ) -> None:
        super().__init__()
        if n_cross < 1:
            raise ValueError(f"n_cross must be >= 1; got {n_cross}")
        if n_self < 0:
            raise ValueError(f"n_self must be >= 0; got {n_self}")
        self.encoder_dim = encoder_dim
        self.predictor_dim = predictor_dim
        self.num_patches = num_patches
        self.max_frames = max_frames
        self.pixel_out_dim = pixel_out_dim
        self.n_cross = n_cross
        self.n_self = n_self

        self.proj_in = nn.Linear(encoder_dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))

        grid = int(math.sqrt(num_patches))
        if grid * grid != num_patches:
            raise ValueError(f"num_patches must be a square; got {num_patches}")
        self.register_buffer(
            "pos_embed_space",
            _sincos_2d_posembed(predictor_dim, grid, cls_token=False).unsqueeze(1),  # (1, 1, N, P)
            persistent=False,
        )
        self.pos_embed_time = nn.Parameter(torch.zeros(1, max_frames, 1, predictor_dim))

        total_depth = n_cross + n_self
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, max(1, total_depth))]
        self.cross_blocks = nn.ModuleList(
            [
                _CrossAttnBlock(predictor_dim, num_heads, mlp_ratio, drop_path=dpr[i])
                for i in range(n_cross)
            ]
        )
        self.self_blocks = nn.ModuleList(
            [
                Block(predictor_dim, num_heads, mlp_ratio, qkv_bias=True, drop_path=dpr[n_cross + i])
                for i in range(n_self)
            ]
        )
        self.norm = nn.LayerNorm(predictor_dim)
        self.proj_out = nn.Linear(predictor_dim, encoder_dim)

        if pixel_out_dim > 0:
            self.pixel_proj = nn.Linear(predictor_dim, pixel_out_dim)
        else:
            self.pixel_proj = None

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_time, std=0.02)
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
        z_ctx: torch.Tensor,
        masks: torch.Tensor,
        T: int,
    ):
        """
        z_ctx: (B, T*N, D_enc) — context-encoder patch tokens.
        masks: (B, T*N) bool   — True at masked positions, constant count per row.
        T:     int              — number of temporal token slices (T_tok).
        """
        B, S, D = z_ctx.shape
        N = self.num_patches
        P = self.predictor_dim
        if S != T * N:
            raise ValueError(f"z_ctx has {S} tokens; expected T*N={T * N}")
        if T > self.max_frames:
            raise ValueError(f"T={T} exceeds max_frames={self.max_frames}")

        n_masked = int(masks[0].sum().item())
        n_visible = T * N - n_masked

        # Project context to predictor dim and add per-position (space+time) pos embeds.
        z = self.proj_in(z_ctx).reshape(B, T, N, P)                        # (B, T, N, P)
        pos = self.pos_embed_space + self.pos_embed_time[:, :T]            # (1, T, N, P)
        z_flat = (z + pos).reshape(B, T * N, P)                            # (B, T*N, P)

        # Build mask-token queries at masked positions: mask_token + pos[mask].
        mask_q_full = (self.mask_token + pos.reshape(1, T * N, P)).expand(B, -1, -1)

        # Batched gather: each row of `masks` has exactly `n_masked` True positions.
        # `tensor[mask]` flattens to (B*n_masked, P) in row-major order; reshape recovers (B, n_masked, P).
        visible_tokens = z_flat[~masks].reshape(B, n_visible, P)           # (B, n_visible, P)
        queries = mask_q_full[masks].reshape(B, n_masked, P)               # (B, n_masked, P)

        # Phase A — extraction: queries cross-attend visible context.
        for blk in self.cross_blocks:
            queries = blk(queries, visible_tokens)

        # Phase B — refinement: joint self-attention over [visible; queries].
        if self.n_self > 0:
            joined = torch.cat([visible_tokens, queries], dim=1)           # (B, T*N, P)
            for blk in self.self_blocks:
                joined = blk(joined)
            joined = self.norm(joined)
            q_out = joined[:, n_visible:]                                  # (B, n_masked, P)
        else:
            q_out = self.norm(queries)

        z_feat = self.proj_out(q_out)                                       # (B, n_masked, D_enc)

        if self.pixel_proj is None:
            return z_feat
        z_pix = self.pixel_proj(q_out)                                      # (B, n_masked, pixel_out_dim)
        return z_feat, z_pix


class VJEPA(nn.Module):
    """Wrapper: per-frame context encoder + EMA target encoder + spacetime
    masked predictor.

    forward(clip) encodes all T frames per-frame with both encoders, samples
    a per-batch block mask over the (T, H, W) tubelet grid, runs the predictor
    on the full spacetime context with masked positions queried by mask_token,
    and computes smooth_L1 between LayerNorm(z_pred) and LayerNorm(z_target)
    only at masked positions.

    EMA update is exposed via ema_step(momentum), called after each optim step.
    """

    def __init__(
        self,
        variant: str = "vit_s_16",
        image_size: int = 224,
        predictor_dim: int = 256,
        predictor_depth: int = 6,
        predictor_heads: int = 8,
        num_frames: int = 4,
        loss_beta: float = 2.0,
        mask_ratio: float = 0.75,
        mask_n_blocks: int = 3,
        tubelet_size: int = 1,
        n_masks: int = 1,
        mask_sampler: str = "block3d",
        loss_type: str = "smooth_l1_lnboth",
        n_long_blocks: int = 2,
        long_scale: float = 0.85,
        n_short_blocks: int = 8,
        short_scale: float = 0.15,
        aspect_min: float = 1.0,
        aspect_max: float = 1.5,
        temporal_tube_fraction: float = 0.0,
        pixel_weight: float = 0.5,
        patch_size: int = 16,
        predictor_type: str = "self",
        predictor_n_cross: int = 3,
        predictor_n_self: int = 2,
    ) -> None:
        super().__init__()
        if num_frames < 2:
            raise ValueError("num_frames must be >= 2")
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0,1); got {mask_ratio}")
        if tubelet_size < 1:
            raise ValueError(f"tubelet_size must be >= 1; got {tubelet_size}")
        if tubelet_size > 1 and num_frames % tubelet_size != 0:
            raise ValueError(f"num_frames {num_frames} must be divisible by tubelet_size {tubelet_size}")
        if n_masks < 1:
            raise ValueError(f"n_masks must be >= 1; got {n_masks}")
        if mask_sampler not in ("block3d", "vjepa", "vjepa_st", "tube_block_v7"):
            raise ValueError(
                f"mask_sampler must be one of 'block3d','vjepa','vjepa_st','tube_block_v7'; got {mask_sampler}"
            )
        if loss_type not in ("smooth_l1_lnboth", "l1_lntarget", "jepa_pixel"):
            raise ValueError(
                f"loss_type must be 'smooth_l1_lnboth', 'l1_lntarget' or 'jepa_pixel'; got {loss_type}"
            )
        if mask_sampler == "block3d" and n_masks != 1:
            raise ValueError("mask_sampler='block3d' supports only n_masks=1")
        if not 0.0 <= temporal_tube_fraction <= 1.0:
            raise ValueError(f"temporal_tube_fraction must be in [0,1]; got {temporal_tube_fraction}")
        if predictor_type not in ("self", "hybrid"):
            raise ValueError(f"predictor_type must be 'self' or 'hybrid'; got {predictor_type}")

        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.t_tokens = num_frames // tubelet_size  # 4 in Tier-1, 2 in Tier-2 with tubelet=2
        self.loss_beta = loss_beta
        self.mask_ratio = mask_ratio
        self.mask_n_blocks = mask_n_blocks
        self.n_masks = n_masks
        self.mask_sampler = mask_sampler
        self.loss_type = loss_type
        self.n_long_blocks = n_long_blocks
        self.long_scale = long_scale
        self.n_short_blocks = n_short_blocks
        self.short_scale = short_scale
        self.aspect_min = aspect_min
        self.aspect_max = aspect_max
        self.temporal_tube_fraction = temporal_tube_fraction
        self.pixel_weight = pixel_weight
        self.patch_size = patch_size

        enc_kwargs = dict(
            variant=variant,
            image_size=image_size,
            tubelet_size=tubelet_size,
            num_frames=num_frames,
        )
        self.context_encoder = VJEPAEncoder(**enc_kwargs)
        # Target shares architecture; copy weights and freeze.
        self.target_encoder = VJEPAEncoder(**enc_kwargs)
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        encoder_dim = self.context_encoder.embed_dim
        num_patches = self.context_encoder.num_patches
        grid = int(math.sqrt(num_patches))
        if grid * grid != num_patches:
            raise ValueError(f"num_patches must be a square; got {num_patches}")
        self.grid = grid
        self.num_patches = num_patches
        # Fixed mask count per sample, shared across the batch so we can batch-gather.
        # In tubelet mode the mask grid spans T_tok temporal slices, not T frames.
        self.n_masked = int(round(mask_ratio * self.t_tokens * num_patches))

        # v7 hybrid: enable pixel head when loss_type asks for it. Output is per
        # tubelet: 3 channels * tubelet_size frames * patch_size^2 spatial.
        self.pixel_out_dim = (
            3 * tubelet_size * patch_size * patch_size if loss_type == "jepa_pixel" else 0
        )

        self.predictor_type = predictor_type
        if predictor_type == "self":
            self.predictor = JEPAPredictor(
                encoder_dim=encoder_dim,
                predictor_dim=predictor_dim,
                depth=predictor_depth,
                num_heads=predictor_heads,
                num_patches=num_patches,
                max_frames=self.t_tokens,
                pixel_out_dim=self.pixel_out_dim,
            )
        else:  # "hybrid"
            self.predictor = HybridCrossSelfPredictor(
                encoder_dim=encoder_dim,
                predictor_dim=predictor_dim,
                n_cross=predictor_n_cross,
                n_self=predictor_n_self,
                num_heads=predictor_heads,
                num_patches=num_patches,
                max_frames=self.t_tokens,
                pixel_out_dim=self.pixel_out_dim,
            )

    def _sample_vjepa_masks(self, B: int, T_tok: int, device: torch.device, M_override: int = -1) -> torch.Tensor:
        """V-JEPA 1 multi-block 3D masks. Returns (B, M, T_tok*N) bool,
        exactly self.n_masked True per row.

        Each of the M masks per clip is the union of `n_long_blocks` long-range
        tubes (spatial area fraction `long_scale / n_long_blocks` of the grid)
        and `n_short_blocks` short-range tubes (spatial area fraction
        `short_scale / n_short_blocks`). All tubes span the full T_tok dim
        (lifted across every time slice) — full-temporal tubes per the V-JEPA
        1 default. Mask coverage is then trimmed/padded to exactly
        `self.n_masked` positions per row so the predictor can batch-gather
        without ragged tensors.

        `M_override` lets the `vjepa_st` dispatcher request a subset of the M
        masks (the rest filled by `_sample_temporal_tube_masks`).
        """
        grid = self.grid
        N = grid * grid
        target = self.n_masked
        M = self.n_masks if M_override < 0 else M_override
        n_long = self.n_long_blocks
        n_short = self.n_short_blocks
        long_area = max(1, int((self.long_scale / max(1, n_long)) * N)) if n_long > 0 else 0
        short_area = max(1, int((self.short_scale / max(1, n_short)) * N)) if n_short > 0 else 0
        a_lo, a_hi = self.aspect_min, self.aspect_max

        masks = torch.zeros(B, M, T_tok, grid, grid, dtype=torch.bool)
        for b in range(B):
            for m in range(M):
                m3d = torch.zeros(T_tok, grid, grid, dtype=torch.bool)
                for area, count in ((long_area, n_long), (short_area, n_short)):
                    if area <= 0 or count <= 0:
                        continue
                    for _ in range(count):
                        aspect = float(torch.empty(1).uniform_(a_lo, a_hi).item())
                        bh = max(1, min(grid, round((area * aspect) ** 0.5)))
                        bw = max(1, min(grid, round(area / max(1, bh))))
                        top = int(torch.randint(0, max(1, grid - bh + 1), (1,)).item())
                        left = int(torch.randint(0, max(1, grid - bw + 1), (1,)).item())
                        m3d[:, top:top + bh, left:left + bw] = True
                masks[b, m] = m3d

        flat = masks.view(B * M, T_tok * N).to(device)
        rand = torch.rand(B * M, T_tok * N, device=device)
        score = flat.float() * 2.0 + rand
        _, sorted_idx = torch.sort(score, dim=1, descending=True)
        out = torch.zeros(B * M, T_tok * N, dtype=torch.bool, device=device)
        rows = torch.arange(B * M, device=device).unsqueeze(1).expand(-1, target)
        out[rows, sorted_idx[:, :target]] = True
        return out.reshape(B, M, T_tok * N)

    def _sample_temporal_tube_masks(self, B: int, T_tok: int, M: int, device: torch.device) -> torch.Tensor:
        """V-JEPA 2-style temporal tubes. For each (b, m), pick a random T_tok
        slice and mark its full spatial extent as True; trim/pad to exactly
        `self.n_masked` True per row. Returns (B, M, T_tok*N) bool.

        At T_tok=2 (our setting), this is binary: mask t=0 OR mask t=1. The
        predictor must reconstruct one frame's-worth of features from the
        other — the cleanest motion supervision possible given num_frames=4.
        Trim/pad ensures a fixed mask budget so we can batch-gather alongside
        the spatial-tube masks in `_sample_st_masks`.
        """
        grid = self.grid
        N = grid * grid
        target = self.n_masked

        masks = torch.zeros(B, M, T_tok, grid, grid, dtype=torch.bool)
        for b in range(B):
            for m in range(M):
                t_idx = int(torch.randint(0, T_tok, (1,)).item())
                masks[b, m, t_idx] = True

        flat = masks.view(B * M, T_tok * N).to(device)
        rand = torch.rand(B * M, T_tok * N, device=device)
        score = flat.float() * 2.0 + rand
        _, sorted_idx = torch.sort(score, dim=1, descending=True)
        out = torch.zeros(B * M, T_tok * N, dtype=torch.bool, device=device)
        rows = torch.arange(B * M, device=device).unsqueeze(1).expand(-1, target)
        out[rows, sorted_idx[:, :target]] = True
        return out.reshape(B, M, T_tok * N)

    def _sample_st_masks(self, B: int, T_tok: int, device: torch.device) -> torch.Tensor:
        """Hybrid V-JEPA-1 spatial tubes + V-JEPA-2 temporal tubes. Splits
        `self.n_masks` into n_spatial = round(M * (1 - temporal_tube_fraction))
        spatial-tube masks and n_temporal = M - n_spatial temporal-tube masks.
        Returns (B, M, T_tok*N) bool.
        """
        M = self.n_masks
        n_spatial = max(0, int(round(M * (1.0 - self.temporal_tube_fraction))))
        n_temporal = M - n_spatial
        parts = []
        if n_spatial > 0:
            parts.append(self._sample_vjepa_masks(B, T_tok, device, M_override=n_spatial))
        if n_temporal > 0:
            parts.append(self._sample_temporal_tube_masks(B, T_tok, n_temporal, device))
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

    def _sample_tube_block_v7(self, B: int, T_tok: int, device: torch.device) -> torch.Tensor:
        """v7 tube-block sampler, multi-mask aware (M = self.n_masks).

        For each (clip, mask) pair, sample one rectangular block in (H_p, W_p)
        covering ~``mask_ratio`` of the spatial grid, then lift it across every
        T_tok slice (canonical V-JEPA tube). With M > 1 the M masks per clip are
        sampled independently — they target different spatial regions, so the
        predictor sees a different visible window each time and the encoder is
        pressured to make features predictable under varied contexts (V-JEPA-1's
        multi-mask sample-efficiency mechanism).

        Returns (B, M, T_tok*N) bool, exactly ``self.n_masked`` True per row.
        """
        grid = self.grid
        N = grid * grid
        target = self.n_masked
        M = self.n_masks
        # Solve bh * bw >= round(mask_ratio * N), keep aspect roughly square.
        spatial_target = max(1, int(round(self.mask_ratio * N)))
        side_lo = int(math.ceil(spatial_target ** 0.5))

        masks = torch.zeros(B, M, T_tok, grid, grid, dtype=torch.bool)
        for b in range(B):
            for m in range(M):
                bh = int(torch.randint(side_lo, grid + 1, (1,)).item())
                bw = int(torch.randint(side_lo, grid + 1, (1,)).item())
                if bh * bw < spatial_target:
                    bw = min(grid, max(bw, math.ceil(spatial_target / bh)))
                top = int(torch.randint(0, grid - bh + 1, (1,)).item())
                left = int(torch.randint(0, grid - bw + 1, (1,)).item())
                spatial_mask = torch.zeros(grid, grid, dtype=torch.bool)
                spatial_mask[top:top + bh, left:left + bw] = True
                masks[b, m] = spatial_mask.unsqueeze(0).expand(T_tok, -1, -1)

        flat = masks.view(B * M, T_tok * N).to(device)
        rand = torch.rand(B * M, T_tok * N, device=device)
        score = flat.float() * 2.0 + rand
        _, sorted_idx = torch.sort(score, dim=1, descending=True)
        out = torch.zeros(B * M, T_tok * N, dtype=torch.bool, device=device)
        rows = torch.arange(B * M, device=device).unsqueeze(1).expand(-1, target)
        out[rows, sorted_idx[:, :target]] = True
        return out.reshape(B, M, T_tok * N)

    def _sample_block_masks(self, B: int, T_tok: int, device: torch.device) -> torch.Tensor:
        """Per-sample 3D block masks over the (T_tok, H, W) token grid, trimmed/
        padded to exactly self.n_masked True positions. Returns (B, T_tok*N).

        3D blocks (Δt × Δh × Δw) sampled in the (T_tok, H, W) grid for both
        Tier-1 (T_tok=num_frames, tubelet=1) and Tier-2 (T_tok=num_frames/tubelet,
        tubelet>1). Partial-temporal blocks force the predictor to reason
        temporally instead of inpainting from neighbouring time slices — required
        for SSv2 where labels are direction/motion-sensitive.
        """
        grid = self.grid
        N = grid * grid
        total = T_tok * N
        target = self.n_masked
        n_blocks = self.mask_n_blocks
        ratio = self.mask_ratio

        masks_grid = torch.zeros(B, T_tok, grid, grid, dtype=torch.bool)
        for b in range(B):
            # 3D blocks: Δt sampled in [1, T_tok], Δh×Δw drawn so the spatial
            # slice has area ~ ratio * N / n_blocks. Stack blocks until the
            # cumulative mask reaches `target`.
            m3d = torch.zeros(T_tok, grid, grid, dtype=torch.bool)
            attempts = 0
            while m3d.sum().item() < target and attempts < 64:
                attempts += 1
                area_frac = ratio / max(1, n_blocks)
                area = max(1, int(area_frac * N))
                aspect = float(torch.empty(1).uniform_(0.5, 2.0).item())
                bh = max(1, min(grid, round((area * aspect) ** 0.5)))
                bw = max(1, min(grid, round(area / max(1, bh))))
                bt = int(torch.randint(1, T_tok + 1, (1,)).item())
                top = int(torch.randint(0, grid - bh + 1, (1,)).item())
                left = int(torch.randint(0, grid - bw + 1, (1,)).item())
                t0 = int(torch.randint(0, T_tok - bt + 1, (1,)).item())
                m3d[t0:t0 + bt, top:top + bh, left:left + bw] = True
            masks_grid[b] = m3d

        masks_flat = masks_grid.view(B, total).to(device)

        # Trim/pad each row to exactly `target` masked positions. Score = mask
        # indicator * 2 + uniform noise; sort descending, take top-target.
        rand = torch.rand(B, total, device=device)
        score = masks_flat.float() * 2.0 + rand
        _, sorted_idx = torch.sort(score, dim=1, descending=True)
        new_masks = torch.zeros(B, total, dtype=torch.bool, device=device)
        rows = torch.arange(B, device=device).unsqueeze(1).expand(-1, target)
        new_masks[rows, sorted_idx[:, :target]] = True
        return new_masks

    def forward(self, clip: torch.Tensor) -> Dict[str, torch.Tensor]:
        """clip: (B, T, 3, H, W). Returns dict(loss, cos_sim, target_std).

        Tier-1: encode each frame independently (fold T into batch).
        Tier-2: encode the whole clip jointly via the 3D-tubelet ViT.
        """
        B, T, C, H, W = clip.shape
        if T != self.num_frames:
            raise ValueError(f"got T={T}, expected num_frames={self.num_frames}")
        N = self.num_patches
        D = self.context_encoder.embed_dim
        T_tok = self.t_tokens
        device = clip.device

        if self.tubelet_size == 1:
            clip_flat = clip.reshape(B * T, C, H, W)
            z_ctx_all = self.context_encoder(clip_flat).reshape(B, T_tok, 1 + N, D)
            z_ctx = z_ctx_all[:, :, 1:, :].reshape(B, T_tok * N, D)          # drop CLS
            with torch.no_grad():
                z_tgt_all = self.target_encoder(clip_flat).reshape(B, T_tok, 1 + N, D)
                z_tgt = z_tgt_all[:, :, 1:, :].reshape(B, T_tok * N, D)
        else:
            z_ctx_all = self.context_encoder(clip)                            # (B, 1+T_tok*N, D)
            z_ctx = z_ctx_all[:, 1:, :]                                       # (B, T_tok*N, D)
            with torch.no_grad():
                z_tgt_all = self.target_encoder(clip)
                z_tgt = z_tgt_all[:, 1:, :]

        # Sample masks: (B, M, T_tok*N) bool, exactly self.n_masked True per row.
        if self.mask_sampler == "vjepa":
            masks_all = self._sample_vjepa_masks(B, T_tok, device)
        elif self.mask_sampler == "vjepa_st":
            masks_all = self._sample_st_masks(B, T_tok, device)
        elif self.mask_sampler == "tube_block_v7":
            masks_all = self._sample_tube_block_v7(B, T_tok, device)
        else:
            masks_all = self._sample_block_masks(B, T_tok, device).unsqueeze(1)

        # Pre-compute target LN once over the full grid for recipes that use
        # global per-token LN; legacy smooth_l1_lnboth LN-aligns the gathered
        # subset per mask instead.
        if self.loss_type in ("l1_lntarget", "jepa_pixel"):
            z_tgt_full_n = F.layer_norm(z_tgt, (D,)).detach()                 # (B, T_tok*N, D)
        else:
            z_tgt_full_n = None

        # Pre-compute pixel target patches (B, T_tok*N, pixel_out_dim) when the
        # hybrid v7 recipe is active. Token order matches Conv3d-tubelet output:
        # [t=0,h=0,w=0], [t=0,h=0,w=1], ..., [t=T_tok-1,h=H_p-1,w=W_p-1].
        if self.loss_type == "jepa_pixel":
            ts = self.tubelet_size
            P = self.patch_size
            H_p = self.grid
            W_p = self.grid
            # (B, T, C, H, W) -> (B, T_tok, ts, C, H_p, P, W_p, P)
            rgb = clip.reshape(B, T_tok, ts, C, H_p, P, W_p, P)
            # -> (B, T_tok, H_p, W_p, ts, P, P, C)
            rgb = rgb.permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()
            # -> (B, T_tok*N, ts*P*P*C)
            rgb = rgb.flatten(4).reshape(B, T_tok * N, ts * P * P * C)
            mean = rgb.mean(-1, keepdim=True)
            var = rgb.var(-1, keepdim=True, unbiased=False)
            rgb_n_full = ((rgb - mean) / (var + 1e-6).sqrt()).detach()
        else:
            rgb_n_full = None

        losses_total, losses_jepa, losses_pix = [], [], []
        cos_sims, target_stds = [], []
        for m in range(self.n_masks):
            mask_m = masks_all[:, m]                                          # (B, T_tok*N)
            pred_out = self.predictor(z_ctx, mask_m, T_tok)
            if self.loss_type == "jepa_pixel":
                z_pred, z_pix = pred_out
            else:
                z_pred = pred_out
                z_pix = None

            if self.loss_type == "smooth_l1_lnboth":
                z_target = z_tgt[mask_m].reshape(B, self.n_masked, D)
                z_pred_n = F.layer_norm(z_pred, (D,))
                z_target_n = F.layer_norm(z_target, (D,))
                loss_jepa_m = F.smooth_l1_loss(z_pred_n, z_target_n.detach(), beta=self.loss_beta)
                with torch.no_grad():
                    cos_m = F.cosine_similarity(z_pred_n, z_target_n, dim=-1).mean()
                    tgt_std_m = z_target_n.std(dim=0).mean()
            else:  # l1_lntarget OR jepa_pixel — both use pre-computed LN target
                z_target_n = z_tgt_full_n[mask_m].reshape(B, self.n_masked, D)
                loss_jepa_m = F.l1_loss(z_pred, z_target_n)
                with torch.no_grad():
                    cos_m = F.cosine_similarity(z_pred, z_target_n, dim=-1).mean()
                    tgt_std_m = z_target_n.std(dim=0).mean()

            if self.loss_type == "jepa_pixel":
                rgb_target = rgb_n_full[mask_m].reshape(B, self.n_masked, self.pixel_out_dim)
                loss_pix_m = F.mse_loss(z_pix, rgb_target)
                loss_total_m = loss_jepa_m + self.pixel_weight * loss_pix_m
            else:
                loss_pix_m = torch.zeros((), device=device)
                loss_total_m = loss_jepa_m

            losses_total.append(loss_total_m)
            losses_jepa.append(loss_jepa_m)
            losses_pix.append(loss_pix_m)
            cos_sims.append(cos_m)
            target_stds.append(tgt_std_m)

        loss = torch.stack(losses_total).mean()
        loss_jepa = torch.stack(losses_jepa).mean().detach()
        loss_pix = torch.stack(losses_pix).mean().detach()
        cos_sim = torch.stack(cos_sims).mean().detach()
        target_std = torch.stack(target_stds).mean().detach()

        return {
            "loss": loss,
            "loss_jepa": loss_jepa,
            "loss_pix": loss_pix,
            "cos_sim": cos_sim,
            "target_std": target_std,
        }

    @torch.no_grad()
    def ema_step(self, momentum: float) -> None:
        """theta_bar <- m * theta_bar + (1 - m) * theta on the target encoder."""
        for pt, pc in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            pt.data.mul_(momentum).add_(pc.data, alpha=1.0 - momentum)
