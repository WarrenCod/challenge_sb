"""
iBOT: Image BERT Pre-training with Online Tokenizer.
Combines DINO (CLS-token self-distillation across multi-crops) with masked
image modeling (patch-level self-distillation at masked positions).

Reference: Zhou et al., 'iBOT: Image BERT Pre-Training with Online Tokenizer'.

Components in this file:
  - iBOTViT       : student/teacher ViT with optional patch masking. Same Block
                    layout as src/models/mae_vit.py so the encoder state_dict
                    drops into ViTMAEEncoder for Stage 2 without renaming.
  - iBOTHead      : 3-layer MLP + L2-norm + weight-norm linear projection.
  - MultiCropEncoder : runs encoder+head on a list of crops, batching
                    same-resolution crops into single forward passes.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block

from models.mae_vit import PatchEmbed, _sincos_2d_posembed


class iBOTViT(nn.Module):
    """ViT encoder used as both student and teacher in iBOT.

    Forward returns (cls_token_out, patch_tokens_out) after final LayerNorm.
    Pos-embed is sized for `img_size` and interpolated on-the-fly for crops at
    other resolutions (local 96×96 vs global 224×224).
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans=3, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # pos_embed is a learnable param (DINO/iBOT use learned, not sincos).
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, drop_path=dpr[i]) for i in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
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

    def interpolate_pos_encoding(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Bilinearly resize the pos-embed to match the (h, w) patch grid.

        x: (B, 1+N, D) where N = h*w. Returns (1, 1+N, D) pos embed for these N patches.
        """
        n = x.shape[1] - 1
        n_orig = self.pos_embed.shape[1] - 1
        if n == n_orig and h * self.patch_size == self.img_size:
            return self.pos_embed

        cls_pe = self.pos_embed[:, :1]
        patch_pe = self.pos_embed[:, 1:]
        d = patch_pe.shape[-1]
        s = int(math.sqrt(n_orig))
        patch_pe = patch_pe.reshape(1, s, s, d).permute(0, 3, 1, 2)  # (1, D, s, s)
        patch_pe = F.interpolate(patch_pe, size=(h, w), mode="bicubic", align_corners=False)
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, h * w, d)
        return torch.cat([cls_pe, patch_pe], dim=1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x:    (B, 3, H, W)
        mask: optional (B, N) bool, True = replace patch token with mask_token.
              Only meaningful for global crops (student side, MIM input).

        Returns (cls_out, patch_out) of shapes (B, D) and (B, N, D).
        """
        b, _, h_img, w_img = x.shape
        x = self.patch_embed(x)  # (B, N, D)
        b_, n, d = x.shape
        h = h_img // self.patch_size
        w = w_img // self.patch_size

        if mask is not None:
            mask_tok = self.mask_token.expand(b_, n, d)
            x = torch.where(mask.unsqueeze(-1), mask_tok, x)

        cls = self.cls_token.expand(b_, -1, -1)
        x = torch.cat([cls, x], dim=1)            # (B, 1+N, D)
        x = x + self.interpolate_pos_encoding(x, h, w)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1:]                  # cls (B, D), patches (B, N, D)

    def encoder_state_dict(self) -> dict:
        """State dict matching the layout of MaskedAutoencoderViT.encoder_state_dict.

        Stage 2 ViTMAEEncoder loads this without renaming.
        """
        keep_prefixes = ("patch_embed.", "cls_token", "blocks.", "norm.")
        full = self.state_dict()
        return {k: v for k, v in full.items() if k.startswith(keep_prefixes)}


class iBOTHead(nn.Module):
    """3-layer MLP + L2 norm + weight-norm linear (DINO/iBOT projection head)."""

    def __init__(
        self,
        in_dim: int = 384,
        out_dim: int = 8192,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        n_layers = max(n_layers, 1)
        if n_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers: List[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False  # DINO trick: weight_g frozen

        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class MultiCropEncoder(nn.Module):
    """Apply (encoder, cls_head, patch_head) to a list of crops of varying resolution.

    Same-resolution crops are concatenated into one forward to save kernel launches.
    Patch-level outputs are only computed for the global crops (first 2 in the list)
    when `apply_mask` is provided.
    """

    def __init__(
        self,
        encoder: iBOTViT,
        cls_head: iBOTHead,
        patch_head: iBOTHead,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.cls_head = cls_head
        self.patch_head = patch_head

    def forward(
        self,
        crops: List[torch.Tensor],
        masks: Optional[List[torch.Tensor]] = None,
        return_patch_for_globals: bool = True,
    ):
        """
        crops: list of length n_global + n_local. Globals first, then locals.
        masks: list of (B, N) bool masks for each global crop (None for locals); only used
               on the student side. Pass None on the teacher side.
        Returns:
            cls_proj:    (n_total * B, out_dim_cls)
            patch_proj:  (n_global * B, N_global, out_dim_patch) or None
        """
        # Group crops by resolution.
        sizes = [c.shape[-1] for c in crops]
        groups: dict = {}  # size -> list of (idx, tensor, mask)
        for i, (c, sz) in enumerate(zip(crops, sizes)):
            groups.setdefault(sz, []).append((i, c, masks[i] if masks is not None else None))

        cls_outs: dict = {}      # idx -> (B, D)
        patch_outs: dict = {}    # idx -> (B, N, D) — only for global, only if return_patch_for_globals

        for sz, items in groups.items():
            tensors = torch.cat([t for _, t, _ in items], dim=0)  # (k*B, 3, sz, sz)
            # Stack masks if any (only relevant for globals).
            if any(m is not None for _, _, m in items):
                mask_stack = torch.cat(
                    [m if m is not None else torch.zeros((items[0][1].shape[0], (sz // self.encoder.patch_size) ** 2), dtype=torch.bool, device=tensors.device) for _, _, m in items],
                    dim=0,
                )
            else:
                mask_stack = None
            cls_o, patch_o = self.encoder(tensors, mask=mask_stack)  # (k*B, D), (k*B, N, D)

            # Split back per crop.
            b = items[0][1].shape[0]
            for j, (idx, _, _) in enumerate(items):
                cls_outs[idx] = cls_o[j * b : (j + 1) * b]
                if return_patch_for_globals and idx < 2:  # first 2 are globals
                    patch_outs[idx] = patch_o[j * b : (j + 1) * b]

        cls_seq = torch.cat([cls_outs[i] for i in range(len(crops))], dim=0)
        cls_proj = self.cls_head(cls_seq)

        patch_proj = None
        if return_patch_for_globals and patch_outs:
            patch_seq = torch.cat([patch_outs[i] for i in sorted(patch_outs)], dim=0)  # (n_global*B, N, D)
            patch_proj = self.patch_head(patch_seq)

        return cls_proj, patch_proj


@torch.no_grad()
def ema_update(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    """teacher_p = momentum * teacher_p + (1 - momentum) * student_p."""
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)
