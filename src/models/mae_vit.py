"""
MAE Vision Transformer (encoder + lightweight decoder + random masking).

Used by Stage 1 of the two-stage pipeline (self-supervised pretraining on
individual frames). The trained encoder state dict is loaded by
``src/models/spatial/vit_mae.py`` for Stage 2 fine-tuning; the decoder
is thrown away.

Reference: He et al., *Masked Autoencoders Are Scalable Vision Learners*,
with timm-style ViT blocks for the encoder and a smaller decoder.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block


def _sincos_2d_posembed(d: int, grid_size: int, cls_token: bool = True) -> torch.Tensor:
    """2D sin-cos positional embedding of shape (1, grid_size**2 + int(cls_token), d).

    Matches the original MAE implementation.
    """
    assert d % 4 == 0, "d must be divisible by 4 for 2D sincos"
    d_half = d // 2
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(grid_w, grid_h, indexing="xy"), dim=0)  # (2, H, W)

    def _pe_1d(pos: torch.Tensor) -> torch.Tensor:
        # pos: (...,), embedding dim d_half
        omega = torch.arange(d_half // 2, dtype=torch.float32) / (d_half / 2.0)
        omega = 1.0 / (10000 ** omega)
        out = pos.reshape(-1)[:, None] * omega[None, :]  # (N, d_half/2)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # (N, d_half)

    pe_w = _pe_1d(grid[0])  # (H*W, d_half)
    pe_h = _pe_1d(grid[1])  # (H*W, d_half)
    pe = torch.cat([pe_w, pe_h], dim=1)  # (H*W, d)

    if cls_token:
        pe = torch.cat([torch.zeros(1, d), pe], dim=0)
    return pe.unsqueeze(0)  # (1, N+1, d)


class PatchEmbed(nn.Module):
    """Conv-based patch embedding, mirrors timm/MAE."""

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 384) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, D, H', W')
        return x.flatten(2).transpose(1, 2)  # (B, N, D)


class MaskedAutoencoderViT(nn.Module):
    """
    Encoder: ViT-S/16-style stack of timm Blocks over (1 + num_kept) tokens.
    Decoder: shallower ViT with mask tokens inserted at masked positions.
    Loss:    MSE on masked patches, per-patch-normalized targets.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        decoder_embed_dim: int = 192,
        decoder_depth: int = 4,
        decoder_num_heads: int = 6,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = True,
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.patch_size = patch_size
        self.in_chans = in_chans

        # Encoder.
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer(
            "pos_embed",
            _sincos_2d_posembed(embed_dim, self.patch_embed.grid_size, cls_token=True),
            persistent=False,
        )
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Decoder.
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.register_buffer(
            "decoder_pos_embed",
            _sincos_2d_posembed(decoder_embed_dim, self.patch_embed.grid_size, cls_token=True),
            persistent=False,
        )
        self.decoder_blocks = nn.ModuleList(
            [
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True)
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * patch_size * in_chans)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        # Patch-embed conv: init like Linear (as MAE paper does).
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

    # ----- patchify / unpatchify for loss -----

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, N, patch_size**2 * C)"""
        p = self.patch_size
        B, C, H, W = imgs.shape
        assert H == W and H % p == 0
        h = w = H // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (B, h, w, p, p, C)
        return x.reshape(B, h * w, p * p * C)

    # ----- masking -----

    def _random_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Per-sample shuffle + keep first (1 - mask_ratio) tokens.
        Returns (x_kept, mask, ids_restore).
          x_kept:      (B, N_keep, D)
          mask:        (B, N)        1 = masked, 0 = kept
          ids_restore: (B, N)        inverse permutation
        """
        B, N, D = x.shape
        n_keep = int(N * (1.0 - self.mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :n_keep]
        x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(B, N, device=x.device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_kept, mask, ids_restore

    # ----- forward passes -----

    def forward_encoder(self, imgs: torch.Tensor):
        x = self.patch_embed(imgs)                # (B, N, D)
        x = x + self.pos_embed[:, 1:, :]          # add patch pos-embeds (no CLS yet)
        x, mask, ids_restore = self._random_masking(x)

        cls = self.cls_token + self.pos_embed[:, :1, :]
        cls = cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)            # (B, 1 + N_keep, D)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        # x: (B, 1 + N_keep, D_enc)
        x = self.decoder_embed(x)  # (B, 1 + N_keep, D_dec)

        B, _, D = x.shape
        N = ids_restore.shape[1]
        n_keep = x.shape[1] - 1
        # pad with mask tokens to length N, then scatter back to original order
        mask_tokens = self.mask_token.expand(B, N - n_keep, -1)
        x_no_cls = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # (B, N, D)
        x_no_cls = torch.gather(
            x_no_cls, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, D),
        )
        x = torch.cat([x[:, :1, :], x_no_cls], dim=1)   # re-prepend CLS
        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)   # (B, 1+N, P*P*C)
        return x[:, 1:, :]         # drop CLS → (B, N, P*P*C)

    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mu = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True, unbiased=False)
            target = (target - mu) / (var + 1e-6).sqrt()
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)          # (B, N)
        return (loss * mask).sum() / mask.sum().clamp(min=1.0)

    def forward(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    # ----- checkpoint helpers -----

    def encoder_state_dict(self) -> dict:
        """Return only encoder-side weights (for Stage 2 loading). Excludes decoder + mask token."""
        keep_prefixes = ("patch_embed.", "cls_token", "blocks.", "norm.")
        full = self.state_dict()
        return {k: v for k, v in full.items() if k.startswith(keep_prefixes)}
