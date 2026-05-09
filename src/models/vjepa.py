"""
V-JEPA Tier-1: multi-frame masked-target prediction.

Self-supervised on T-frame clips. Two ViT-S/16 encoders (context f_theta and
EMA target f_theta_bar) share architecture, applied per-frame so the encoder
stays MAE/iBOT-compatible for Stage 2. The predictor consumes the full
spacetime grid of context patch tokens (B, T*N, D), replaces masked positions
with a learnable mask query (mask_token + space-pos + time-pos), runs joint
self-attention, and predicts the EMA target encoder's patch features only at
the masked positions.

Block masking: per sample, mask K rectangles in the (H, W) grid lifted across
all T frames. The same spatial holes appear in every frame, so the predictor
must use spatial neighbours to fill them — and where holes coincide with
moving content, temporal redundancy across frames carries the signal. Mask
ratio ~0.75 forces the encoder to produce dense, locally-informative features
(MAE-style pressure) rather than collapse to "next frame ≈ this frame".

Loss: smooth_L1 in feature space at masked positions, scale-aligned via
parameter-free LayerNorm on both prediction and target.

The saved EMA-encoder dump still uses the same key layout as MAE / iBOT
(`encoder_state_dict`), so Stage 2 loads it via the existing ViTMAEEncoder
without renaming.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block

from models.mae_vit import PatchEmbed, _sincos_2d_posembed, _sincos_3d_posembed


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
    ) -> None:
        super().__init__()
        self.encoder_dim = encoder_dim
        self.predictor_dim = predictor_dim
        self.num_patches = num_patches
        self.max_frames = max_frames

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
    ) -> torch.Tensor:
        """
        z_ctx: (B, T*N, D_enc) — context-encoder patch tokens, all positions.
        masks: (B, T*N)  bool  — True at positions to predict (constant count per row).
        T:     int             — number of frames.

        Returns: (B, n_masked, D_enc) — predictions at the masked positions only.
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
        z = self.norm(z)
        z = self.proj_out(z)                                                  # (B, T*N, D_enc)

        # Gather predictions at masked positions only. masks has fixed count
        # per row, so the reshape is well-defined.
        n_masked = int(masks[0].sum().item())
        return z[masks].reshape(B, n_masked, D)


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

        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.t_tokens = num_frames // tubelet_size  # 4 in Tier-1, 2 in Tier-2 with tubelet=2
        self.loss_beta = loss_beta
        self.mask_ratio = mask_ratio
        self.mask_n_blocks = mask_n_blocks

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

        self.predictor = JEPAPredictor(
            encoder_dim=encoder_dim,
            predictor_dim=predictor_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
            num_patches=num_patches,
            max_frames=self.t_tokens,
        )

    def _sample_block_masks(self, B: int, T_tok: int, device: torch.device) -> torch.Tensor:
        """Per-sample block masks over the (T_tok, H, W) token grid, trimmed/
        padded to exactly self.n_masked True positions. Returns (B, T_tok*N).

        Tier-1 (tubelet=1): 2D rectangles in (H, W) lifted across all T frames
        — same spatial holes appear in every frame. Forces spatial inpainting,
        but a holey frame can be filled trivially from neighbouring frames.

        Tier-2 (tubelet>1): 3D rectangles (Δt × Δh × Δw) sampled in the
        (T_tok, H, W) grid — partial-temporal blocks force the predictor to
        reason temporally instead of inpainting from the unmasked time slice.
        """
        grid = self.grid
        N = grid * grid
        total = T_tok * N
        target = self.n_masked
        n_blocks = self.mask_n_blocks
        ratio = self.mask_ratio

        masks_grid = torch.zeros(B, T_tok, grid, grid, dtype=torch.bool)
        for b in range(B):
            if self.tubelet_size == 1:
                # 2D mask lifted across all temporal positions.
                mask_hw = torch.zeros(grid, grid, dtype=torch.bool)
                attempts = 0
                while mask_hw.sum().item() * T_tok < target and attempts < 32:
                    attempts += 1
                    area_frac = ratio / max(1, n_blocks)
                    area = max(1, int(area_frac * N))
                    aspect = float(torch.empty(1).uniform_(0.5, 2.0).item())
                    bh = max(1, min(grid, round((area * aspect) ** 0.5)))
                    bw = max(1, min(grid, round(area / max(1, bh))))
                    top = int(torch.randint(0, grid - bh + 1, (1,)).item())
                    left = int(torch.randint(0, grid - bw + 1, (1,)).item())
                    mask_hw[top:top + bh, left:left + bw] = True
                masks_grid[b] = mask_hw.unsqueeze(0).expand(T_tok, -1, -1)
            else:
                # 3D blocks: Δt sampled in [1, T_tok], Δh×Δw drawn so the
                # spatial slice has area ~ ratio * N / n_blocks. Stack blocks
                # until the cumulative mask reaches `target`.
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

        # Sample block masks: (B, T_tok*N), exactly self.n_masked True per row.
        masks = self._sample_block_masks(B, T_tok, device)

        z_pred = self.predictor(z_ctx, masks, T_tok)                          # (B, n_masked, D)
        z_target = z_tgt[masks].reshape(B, self.n_masked, D)                  # (B, n_masked, D)

        # Parameter-free LayerNorm on both sides for scale-invariant matching.
        z_pred_n = F.layer_norm(z_pred, (D,))
        z_target_n = F.layer_norm(z_target, (D,))

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
