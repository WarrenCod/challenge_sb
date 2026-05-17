"""Shared ViT building blocks: patch embedding + sincos positional encodings.

Used by the V-JEPA pretraining and Stage-2 fine-tuning paths. Originally
lived inside ``models/mae_vit.py``; extracted here so we can drop MAE
without losing the geometry primitives.
"""

from __future__ import annotations

import torch
import torch.nn as nn


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


def _sincos_3d_posembed(d: int, t_tokens: int, grid_size: int, cls_token: bool = True) -> torch.Tensor:
    """3D sin-cos pos-embed, factorized as 2D-spatial + 1D-temporal (summed).

    Output: (1, cls + t_tokens * grid_size**2, d). Order: temporal-major
    (slice 0 patches, slice 1 patches, ...). VideoMAE / V-JEPA-2 style.
    """
    assert d % 4 == 0, "d must be divisible by 4"
    spat = _sincos_2d_posembed(d, grid_size, cls_token=False).squeeze(0)  # (N_sp, d)

    pos_t = torch.arange(t_tokens, dtype=torch.float32)
    omega = torch.arange(d // 2, dtype=torch.float32) / (d / 2.0)
    omega = 1.0 / (10000 ** omega)
    out = pos_t[:, None] * omega[None, :]  # (T, d/2)
    temp = torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # (T, d)

    n_sp = grid_size * grid_size
    spat_3d = spat.unsqueeze(0).expand(t_tokens, -1, -1)        # (T, N_sp, d)
    temp_3d = temp.unsqueeze(1).expand(-1, n_sp, -1)             # (T, N_sp, d)
    pe = (spat_3d + temp_3d).reshape(t_tokens * n_sp, d)         # (T*N_sp, d)

    if cls_token:
        pe = torch.cat([torch.zeros(1, d), pe], dim=0)
    return pe.unsqueeze(0)  # (1, cls + T*N_sp, d)


class PatchEmbed(nn.Module):
    """Conv-based 2D patch embedding (timm/MAE style)."""

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
