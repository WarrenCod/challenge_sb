"""V-JEPA 2 classifier with multi-block attentive probe head.

Replaces the mean-pool over 512 space-time tokens used in vjepa2.py with
the V-JEPA 2-style attentive probe: K learnable queries cross-attend over
encoder tokens through N blocks of {cross-attn -> self-attn -> MLP}, then
mean-pool the K queries -> Linear.

Backbone reused from vjepa2.py (``VJepa2EncoderWrap``) untouched, so LLRD
(via ``model.spatial.ordered_layers()``) keeps the same scope. The probe
lives outside ``.spatial`` and is collected into the existing head param
group; ``training.head_lr_mult`` (plumbed in train.py / utils.py) scales it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vjepa2 import VJepa2EncoderWrap


class _CrossAttn(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q: (B, K, d) -- learnable queries (post-LN)
        # kv: (B, N, d) -- encoder tokens (post-LN)
        B, K, D = q.shape
        N = kv.shape[1]
        H, Dh = self.num_heads, self.head_dim
        q = self.q_proj(q).reshape(B, K, H, Dh).transpose(1, 2)            # (B, H, K, Dh)
        kv = self.kv_proj(kv).reshape(B, N, 2, H, Dh).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]                                                 # (B, H, N, Dh)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)     # (B, H, K, Dh)
        out = out.transpose(1, 2).reshape(B, K, D)
        return self.out_proj(out)


class _SelfAttn(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, K, D = x.shape
        H, Dh = self.num_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, K, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                                    # (B, H, K, Dh)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(1, 2).reshape(B, K, D)
        return self.out_proj(out)


class _MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int) -> None:
        super().__init__()
        hidden = dim * mlp_ratio
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _ProbeBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.norm_q1 = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = _CrossAttn(dim, num_heads)
        self.norm_q2 = nn.LayerNorm(dim)
        self.self_attn = _SelfAttn(dim, num_heads)
        self.norm_q3 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, mlp_ratio)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q = q + self.cross_attn(self.norm_q1(q), self.norm_kv(kv))
        q = q + self.self_attn(self.norm_q2(q))
        q = q + self.mlp(self.norm_q3(q))
        return q


class AttnProbe(nn.Module):
    """V-JEPA 2-style attentive probe: K queries x N blocks -> mean-pool -> (B, d)."""

    def __init__(
        self,
        dim: int,
        n_blocks: int = 2,
        n_queries: int = 16,
        n_heads: int = 16,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        # Param literally named `queries` so utils.build_llrd_param_groups'
        # `no_decay_names` exemption picks it up (it's 2-D so would otherwise be decayed).
        self.queries = nn.Parameter(torch.empty(n_queries, dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.blocks = nn.ModuleList(
            [_ProbeBlock(dim, n_heads, mlp_ratio) for _ in range(n_blocks)]
        )
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, N, d) -> pooled: (B, d)
        B = tokens.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)                     # (B, K, d)
        for blk in self.blocks:
            q = blk(q, tokens)
        return self.norm_out(q).mean(dim=1)


class VJepa2ProbeClassifier(nn.Module):
    """V-JEPA 2 encoder + AttnProbe + Linear head."""

    def __init__(
        self,
        num_classes: int,
        hf_id: str = "facebook/vjepa2-vitl-fpc64-256",
        pretrained: bool = True,
        num_frames: int = 4,
        image_size: int = 256,
        drop_path: float = 0.2,
        head_dropout: float = 0.1,
        probe_n_blocks: int = 2,
        probe_n_queries: int = 16,
        probe_n_heads: int = 16,
        probe_mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.spatial = VJepa2EncoderWrap(
            hf_id=hf_id,
            pretrained=pretrained,
            num_frames=num_frames,
            image_size=image_size,
            drop_path=drop_path,
        )
        d = self.spatial.embed_dim
        self.probe = AttnProbe(
            dim=d,
            n_blocks=probe_n_blocks,
            n_queries=probe_n_queries,
            n_heads=probe_n_heads,
            mlp_ratio=probe_mlp_ratio,
        )
        self.head_drop = nn.Dropout(head_dropout)
        self.head = nn.Linear(d, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        tokens = self.spatial(video)                # (B, N, d)
        pooled = self.probe(tokens)                 # (B, d)
        return self.head(self.head_drop(pooled))
