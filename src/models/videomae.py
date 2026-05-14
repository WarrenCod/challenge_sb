"""
VideoMAE: spatio-temporal masked autoencoder for Stage-1 SSL pretraining.

Tubelet embedding (joint space-time patches) + asymmetric encoder/decoder + tube
masking + per-tubelet normalized-pixel MSE.

The encoder is reused at Stage 2 (decoder is thrown away). Two classes:

  * ``VideoMAEModel``    — full SSL model (encoder + decoder + loss).
  * ``VideoMAEEncoder``  — encoder-only wrapper for inference / Stage 2.
                           Returns spatial-mean-pooled tokens of shape (B, T', D)
                           so the existing modular temporal heads plug in.

Reference: Tong et al., *VideoMAE: Masked Autoencoders Are Data-Efficient Learners
for Self-Supervised Video Pre-Training*.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block


def _sincos_1d(d: int, length: int) -> torch.Tensor:
    """1-D sin-cos positional embedding of shape (length, d)."""
    assert d % 2 == 0, "1d-sincos requires even dim"
    pos = torch.arange(length, dtype=torch.float32)
    omega = torch.arange(d // 2, dtype=torch.float32) / (d / 2.0)
    omega = 1.0 / (10000 ** omega)
    out = pos[:, None] * omega[None, :]
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # (length, d)


def _sincos_2d(d: int, grid_size: int) -> torch.Tensor:
    """2-D sin-cos positional embedding of shape (grid_size**2, d)."""
    assert d % 4 == 0, "2d-sincos requires dim divisible by 4"
    d_half = d // 2
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(grid_w, grid_h, indexing="xy"), dim=0)  # (2, H, W)

    def _pe_1d(pos: torch.Tensor) -> torch.Tensor:
        omega = torch.arange(d_half // 2, dtype=torch.float32) / (d_half / 2.0)
        omega = 1.0 / (10000 ** omega)
        out = pos.reshape(-1)[:, None] * omega[None, :]
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    pe_w = _pe_1d(grid[0])
    pe_h = _pe_1d(grid[1])
    return torch.cat([pe_w, pe_h], dim=1)  # (H*W, d)


def _sincos_3d(d: int, t_grid: int, hw_grid: int) -> torch.Tensor:
    """Factorized 3-D sin-cos pos-embed of shape (1, t_grid * hw_grid**2, d).

    Splits d in half: first half = 1-D sincos over time, second half = 2-D sincos
    over (h, w). For each (t, h, w) token the result is concat[time_emb_t, hw_emb_{h,w}].
    """
    assert d % 2 == 0, "3d-sincos requires even dim"
    d_t = d // 2
    d_s = d - d_t
    pe_t = _sincos_1d(d_t, t_grid)              # (T', d_t)
    pe_s = _sincos_2d(d_s, hw_grid)             # (H'*W', d_s)
    n_s = pe_s.size(0)
    # Cartesian product: broadcast t over space, space over time.
    pe_t_full = pe_t[:, None, :].expand(t_grid, n_s, d_t).reshape(t_grid * n_s, d_t)
    pe_s_full = pe_s[None, :, :].expand(t_grid, n_s, d_s).reshape(t_grid * n_s, d_s)
    pe = torch.cat([pe_t_full, pe_s_full], dim=1)  # (T'*H'*W', d)
    return pe.unsqueeze(0)                          # (1, N, d)


class TubeletEmbed(nn.Module):
    """Joint space-time Conv3d patch. ``(B, T, C, H, W) -> (B, N=T'*H'*W', D)``."""

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
        self.hw_grid = img_size // tubelet_size
        self.num_tokens = self.t_grid * self.hw_grid * self.hw_grid
        self.tubelet_time = tubelet_time
        self.tubelet_size = tubelet_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(tubelet_time, tubelet_size, tubelet_size),
            stride=(tubelet_time, tubelet_size, tubelet_size),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # (B, T, C, H, W) -> (B, C, T, H, W) for Conv3d.
        x = video.permute(0, 2, 1, 3, 4)
        x = self.proj(x)                       # (B, D, T', H', W')
        return x.flatten(2).transpose(1, 2)    # (B, N, D)


def _tube_masking(
    batch_size: int,
    t_grid: int,
    hw_count: int,
    mask_ratio: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tube masking: choose the *same* spatial subset across all t-slots.

    Args:
        batch_size: B.
        t_grid: T'.
        hw_count: H' * W'.
        mask_ratio: fraction of tubelets to mask (over total T' * H' * W').

    Returns:
        ids_keep:    (B, n_keep_total)    flat indices into the (T', H'*W') grid.
        mask:        (B, T' * H'*W')      1 = masked, 0 = kept.

    Indexing convention matches TubeletEmbed: flat index = t * H'*W' + s, where
    s is the spatial position. So masking spatial positions {s_i} across all t
    produces the keep indices [t * hw_count + s : for s in keep_spatial, for t in 0..T'-1].
    """
    n_spatial = hw_count
    n_keep_spatial = max(1, int(round(n_spatial * (1.0 - mask_ratio))))

    # Per-sample random spatial subset.
    noise = torch.rand(batch_size, n_spatial, device=device)
    ids_shuffle_spatial = torch.argsort(noise, dim=1)
    keep_spatial = ids_shuffle_spatial[:, :n_keep_spatial]    # (B, n_keep_spatial)

    # Build full keep ids across all t-slots: flat = t * hw_count + s.
    t_offsets = torch.arange(t_grid, device=device) * n_spatial          # (T',)
    # (B, T', n_keep_spatial) = keep_spatial broadcast + t_offsets.
    ids_keep_3d = keep_spatial.unsqueeze(1) + t_offsets.view(1, t_grid, 1)
    ids_keep = ids_keep_3d.reshape(batch_size, t_grid * n_keep_spatial)  # (B, n_keep_total)

    # Build mask: ones, then zero out kept positions.
    mask = torch.ones(batch_size, t_grid * n_spatial, device=device)
    mask.scatter_(1, ids_keep, 0.0)
    return ids_keep, mask


class VideoMAEModel(nn.Module):
    """Encoder + lightweight decoder + tube masking + normalized-pixel MSE loss."""

    def __init__(
        self,
        num_frames: int = 4,
        img_size: int = 224,
        tubelet_time: int = 2,
        tubelet_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        decoder_embed_dim: int = 192,
        decoder_depth: int = 4,
        decoder_num_heads: int = 6,
        mask_ratio: float = 0.90,
        norm_pix_loss: bool = True,
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.num_frames = num_frames
        self.img_size = img_size
        self.tubelet_time = tubelet_time
        self.tubelet_size = tubelet_size
        self.in_chans = in_chans

        # Encoder.
        self.patch_embed = TubeletEmbed(
            num_frames=num_frames,
            img_size=img_size,
            tubelet_time=tubelet_time,
            tubelet_size=tubelet_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.t_grid = self.patch_embed.t_grid
        self.hw_grid = self.patch_embed.hw_grid
        self.hw_count = self.hw_grid * self.hw_grid
        self.num_tokens = self.patch_embed.num_tokens
        self.embed_dim = embed_dim

        self.register_buffer(
            "pos_embed",
            _sincos_3d(embed_dim, self.t_grid, self.hw_grid),
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
            _sincos_3d(decoder_embed_dim, self.t_grid, self.hw_grid),
            persistent=False,
        )
        self.decoder_blocks = nn.ModuleList(
            [
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True)
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        # Predicts per-tubelet flattened pixels: t * p * p * C channels.
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, tubelet_time * tubelet_size * tubelet_size * in_chans
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.mask_token, std=0.02)
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

    # ---- patchify for loss target --------------------------------------------------

    def patchify(self, video: torch.Tensor) -> torch.Tensor:
        """(B, T, C, H, W) -> (B, N=T'*H'*W', tubelet_time * tubelet_size**2 * C).

        Tubelets are flattened in the (t, h, w) → row-major order so they line up
        with the Conv3d output of TubeletEmbed (same as `flatten(2).transpose`).
        """
        B, T, C, H, W = video.shape
        t = self.tubelet_time
        p = self.tubelet_size
        assert T == self.num_frames and H == W == self.img_size
        t_grid = T // t
        s_grid = H // p

        # (B, T, C, H, W) -> (B, t_grid, t, C, s_grid, p, s_grid, p)
        x = video.reshape(B, t_grid, t, C, s_grid, p, s_grid, p)
        # -> (B, t_grid, s_grid, s_grid, t, p, p, C)
        x = x.permute(0, 1, 4, 6, 2, 5, 7, 3)
        # -> (B, t_grid * s_grid * s_grid, t * p * p * C)
        return x.reshape(B, t_grid * s_grid * s_grid, t * p * p * C)

    # ---- forward passes ------------------------------------------------------------

    def forward_encoder(self, video: torch.Tensor):
        """Returns (latent, mask, ids_keep) where latent is (B, n_keep, D)."""
        x = self.patch_embed(video)             # (B, N, D)
        x = x + self.pos_embed                  # add 3-D pos-embed before masking
        ids_keep, mask = _tube_masking(
            batch_size=x.size(0),
            t_grid=self.t_grid,
            hw_count=self.hw_count,
            mask_ratio=self.mask_ratio,
            device=x.device,
        )
        D = x.size(-1)
        x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        for blk in self.blocks:
            x_kept = blk(x_kept)
        x_kept = self.norm(x_kept)
        return x_kept, mask, ids_keep

    def forward_decoder(self, x_kept: torch.Tensor, ids_keep: torch.Tensor) -> torch.Tensor:
        """Returns predictions over all N tokens: (B, N, t*p*p*C)."""
        B = x_kept.size(0)
        N = self.num_tokens
        D = self.decoder_embed.out_features

        x = self.decoder_embed(x_kept)          # (B, n_keep, D_dec)

        # Scatter visible tokens to their original positions; fill the rest with mask_token.
        # Match x's dtype so this works under bf16 autocast (mask_token is fp32 param).
        full = self.mask_token.expand(B, N, -1).to(x.dtype).clone()  # (B, N, D_dec)
        full.scatter_(
            1,
            ids_keep.unsqueeze(-1).expand(-1, -1, D),
            x,
        )
        full = full + self.decoder_pos_embed.to(x.dtype)  # 3-D pos-embed at decoder side

        for blk in self.decoder_blocks:
            full = blk(full)
        full = self.decoder_norm(full)
        return self.decoder_pred(full)                   # (B, N, t*p*p*C)

    def forward_loss(
        self,
        video: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        target = self.patchify(video)                    # (B, N, t*p*p*C)
        if self.norm_pix_loss:
            mu = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True, unbiased=False)
            target = (target - mu) / (var + 1e-6).sqrt()
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)                         # (B, N)
        return (loss * mask).sum() / mask.sum().clamp(min=1.0)

    def forward(self, video: torch.Tensor):
        latent, mask, ids_keep = self.forward_encoder(video)
        pred = self.forward_decoder(latent, ids_keep)
        loss = self.forward_loss(video, pred, mask)
        return loss, pred, mask

    # ---- checkpoint helpers --------------------------------------------------------

    def encoder_state_dict(self) -> dict:
        """Encoder-side weights only (patch_embed + blocks + norm). For Stage 2 loading."""
        keep_prefixes = ("patch_embed.", "blocks.", "norm.")
        full = self.state_dict()
        return {k: v for k, v in full.items() if k.startswith(keep_prefixes)}


class VideoMAEEncoder(nn.Module):
    """Encoder-only wrapper used at inference (probe) and Stage 2.

    Loads encoder weights from a Stage-1 checkpoint (key ``encoder_state_dict``)
    and produces space-time tokens. Two output modes:

    * ``forward(video)`` → ``(B, T', D)`` — spatial mean over (H', W') per tubelet.
                          Fits the existing modular ``SpatialEncoder`` contract.
    * ``forward_features(video)`` → ``(B, T'*H'*W', D)`` — full token grid for the probe.

    Implements ``ordered_layers()`` so ``build_llrd_param_groups`` works.
    """

    def __init__(
        self,
        num_frames: int = 4,
        img_size: int = 224,
        tubelet_time: int = 2,
        tubelet_size: int = 16,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_path: "float | list[float]" = 0.0,
        checkpoint_path: str | None = None,
    ) -> None:
        super().__init__()
        self.out_dim = embed_dim
        self.embed_dim = embed_dim
        self.patch_embed = TubeletEmbed(
            num_frames=num_frames,
            img_size=img_size,
            tubelet_time=tubelet_time,
            tubelet_size=tubelet_size,
            in_chans=3,
            embed_dim=embed_dim,
        )
        self.t_grid = self.patch_embed.t_grid
        self.hw_grid = self.patch_embed.hw_grid
        self.hw_count = self.hw_grid * self.hw_grid
        self.num_tokens = self.patch_embed.num_tokens

        self.register_buffer(
            "pos_embed",
            _sincos_3d(embed_dim, self.t_grid, self.hw_grid),
            persistent=False,
        )
        if isinstance(drop_path, (list, tuple)):
            if len(drop_path) != depth:
                raise ValueError(
                    f"drop_path list length {len(drop_path)} != depth {depth}"
                )
            dp_rates = [float(x) for x in drop_path]
        else:
            dp_rates = [float(drop_path)] * depth
        self.blocks = nn.ModuleList(
            [
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, drop_path=dp_rates[i])
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Match the init convention of VideoMAEModel for shape-compatible warm-up.
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

        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, path: str) -> None:
        from pathlib import Path
        ck = torch.load(Path(path).resolve(), map_location="cpu", weights_only=False)
        state = ck.get("encoder_state_dict", ck)
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing:
            print(f"[videomae] missing keys: {len(missing)} (e.g. {missing[:3]})")
        if unexpected:
            print(f"[videomae] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
        print(f"[videomae] loaded encoder from {path}")

    def forward_features(self, video: torch.Tensor) -> torch.Tensor:
        """All space-time tokens. (B, T, C, H, W) -> (B, T'*H'*W', D)."""
        x = self.patch_embed(video)        # (B, N, D)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Spatial mean per tubelet. (B, T, C, H, W) -> (B, T', D)."""
        tokens = self.forward_features(video)                       # (B, T'*HW, D)
        B, _, D = tokens.shape
        tokens = tokens.view(B, self.t_grid, self.hw_count, D)
        return tokens.mean(dim=2)                                   # (B, T', D)

    def ordered_layers(self):
        """Shallow → deep iterator for layer-wise LR decay."""
        yield ("embed", nn.ModuleList([self.patch_embed]))
        for i, blk in enumerate(self.blocks):
            yield (f"block_{i}", blk)
        yield ("norm", self.norm)
