"""
LaplacianSwinTransformer: Swin-T backbone enhanced with MEGANet's Laplacian
edge-guided attention (EGA) and interleaved top-down decoder.

Architecture (strictly follows MEGANet's decoder+EGA pattern):
  Encoder:  s0(96) → s1(192) → s2(384) → s3(768)   [Swin-T 4 stages]
  Laplacian: shared edge map from input image

  Decoder (top-down, interleaved with EGA):
    s3 → up_s3 → d3 → pred3          (deepest, no EGA)
    EGA(edge, s2, pred3) → ega2
    d3 + ega2 → up_s2 → d2 → pred2
    EGA(edge, s1, pred2) → ega1
    d2 + ega1 → up_s1 → d1 → pred1
    EGA(edge, s0, pred1) → ega0
    d1 + ega0 → up_s0 → d0

  Output for FPN: [ega1(192), ega2(384), s3(768)]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale

from mmdet3d.compat import BaseModule, BACKBONES
from .swin import SwinTransformer
from .laplacian_ega import (
    make_laplace_pyramid,
    EGA,
    Conv,
    Up,
    Out,
)


@BACKBONES.register_module()
class LaplacianSwinTransformer(BaseModule):
    """Swin-T backbone with Laplacian EGA decoder.

    Args:
        swin_cfg (dict): Configuration for the inner SwinTransformer.
            Must include ``out_indices=[0, 1, 2, 3]`` to output all 4 stages.
        laplacian_level (int): Number of Laplacian pyramid levels. Default: 5.
        laplacian_index (int): Which pyramid level to use as edge feature.
            Default: 1 (2nd level, matching MEGANet).
    """

    def __init__(self, swin_cfg, laplacian_level=5, laplacian_index=1, **kwargs):
        super(LaplacianSwinTransformer, self).__init__()

        # Build inner Swin-T encoder
        self.swin = SwinTransformer(**swin_cfg)

        self.laplacian_level = laplacian_level
        self.laplacian_index = laplacian_index

        # Channel dimensions for each Swin-T stage
        embed_dims = swin_cfg.get('embed_dims', 96)
        ch = [embed_dims * (2 ** i) for i in range(4)]  # [96, 192, 384, 768]

        # --- Decoder modules (following MEGANet's Up / Out pattern) ---

        # Deepest stage: conv + upsample (no concat, like MEGANet's up5)
        self.up_s3 = nn.Sequential(
            Conv(ch[3], ch[3]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        # Merge stages: concat(d_prev, ega) → conv → upsample
        self.up_s2 = Up(ch[3] + ch[2], ch[2])   # 1152 → 384
        self.up_s1 = Up(ch[2] + ch[1], ch[1])   # 576  → 192
        self.up_s0 = Up(ch[1] + ch[0], ch[0])   # 288  → 96

        # Prediction heads (1-channel output, like MEGANet's Out)
        self.out3 = Out(ch[3], 1)
        self.out2 = Out(ch[2], 1)
        self.out1 = Out(ch[1], 1)

        # EGA modules — full 3-input (edge_feature, encoder_feat, pred)
        self.ega2 = EGA(ch[2])  # 384
        self.ega1 = EGA(ch[1])  # 192
        self.ega0 = EGA(ch[0])  # 96

    def forward(self, x):
        # ---- 1. Laplacian edge feature (shared across all stages) ----
        with torch.no_grad():
            grayscale = rgb_to_grayscale(x.float())  # (B*N, 1, H, W)
            edge_pyramid = make_laplace_pyramid(grayscale, self.laplacian_level, 1)
            edge_feature = edge_pyramid[self.laplacian_index]  # (B*N, 1, H', W')

        # ---- 2. Swin-T encoder (4 stages) ----
        swin_outs = self.swin(x)  # list of 4 tensors: [s0, s1, s2, s3]
        s0, s1, s2, s3 = swin_outs

        # ---- 3. Interleaved decoder + EGA (strictly following MEGANet) ----

        # Stage 3 (deepest) — no EGA, bootstrap decoder
        d3 = self.up_s3(s3)               # (B*N, 768, H/16, W/16)
        pred3 = self.out3(d3)             # (B*N, 1,   H/16, W/16)

        # Stage 2 — EGA guided by pred3
        ega2_out = self.ega2(edge_feature, s2, pred3)  # (B*N, 384, H/16, W/16)
        d2 = self.up_s2(d3, ega2_out)                  # (B*N, 384, H/8,  W/8)
        pred2 = self.out2(d2)                           # (B*N, 1,   H/8,  W/8)

        # Stage 1 — EGA guided by pred2
        ega1_out = self.ega1(edge_feature, s1, pred2)  # (B*N, 192, H/8,  W/8)
        d1 = self.up_s1(d2, ega1_out)                  # (B*N, 192, H/4,  W/4)
        pred1 = self.out1(d1)                           # (B*N, 1,   H/4,  W/4)

        # Stage 0 — EGA guided by pred1
        ega0_out = self.ega0(edge_feature, s0, pred1)  # (B*N, 96,  H/4,  W/4)
        d0 = self.up_s0(d1, ega0_out)                  # (B*N, 96,  H/2,  W/2)

        # ---- 4. Multi-scale output for FPN [192, 384, 768] ----
        return (ega1_out, ega2_out, s3)
