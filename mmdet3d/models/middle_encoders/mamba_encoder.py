"""
MambaMiddleEncoder: Replaces DynamicVFE + SparseEncoder with a Mamba-based
BEV encoder for the point cloud branch.

Pipeline:
  Raw Points -> Pillarization (BEV grid) -> Conv1d Encoder (per-pillar)
  -> Hilbert Serialization -> Bidirectional Mamba Blocks -> BEV Features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models.builder import MIDDLE_ENCODERS
from .pointmamba_utils import (
    MixerModel,
    init_OrderScale,
    apply_OrderScale,
    serialization_func,
)


class PillarEncoder(nn.Module):
    """Per-pillar point feature encoder using Linear + max-pool.

    Uses nn.Linear instead of nn.Conv1d(kernel_size=1) to avoid cuDNN
    backward issues with very large batch dimensions (tens of thousands
    of pillars treated as batch).
    """

    def __init__(self, in_channels, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_linear1 = nn.Linear(in_channels, 128)
        self.first_bn1 = nn.BatchNorm1d(128)
        self.first_linear2 = nn.Linear(128, 256)

        self.second_linear1 = nn.Linear(512, 512)
        self.second_bn1 = nn.BatchNorm1d(512)
        self.second_linear2 = nn.Linear(512, encoder_channel)

    def forward(self, pillar_features):
        """
        Args:
            pillar_features: (K, P, C) — K pillars, P points each, C channels
        Returns:
            (K, encoder_channel)
        """
        # pillar_features: (K, P, C)
        # first block: Linear(C->128) + BN + ReLU + Linear(128->256)
        x = self.first_linear1(pillar_features)      # (K, P, 128)
        x = self.first_bn1(x.transpose(1, 2)).transpose(1, 2)  # BN over channel dim
        x = F.relu(x)
        x = self.first_linear2(x)                    # (K, P, 256)

        # global max pool
        feature_global = torch.max(x, dim=1, keepdim=True)[0]  # (K, 1, 256)
        # concat global + per-point: (K, P, 512)
        feature = torch.cat([feature_global.expand(-1, x.shape[1], -1), x], dim=2)

        # second block: Linear(512->512) + BN + ReLU + Linear(512->encoder_channel)
        feature = self.second_linear1(feature)        # (K, P, 512)
        feature = self.second_bn1(feature.transpose(1, 2)).transpose(1, 2)
        feature = F.relu(feature)
        feature = self.second_linear2(feature)        # (K, P, encoder_channel)

        # global max pool -> (K, encoder_channel)
        feature_global = torch.max(feature, dim=1, keepdim=False)[0]
        return feature_global


@MIDDLE_ENCODERS.register_module()
class MambaMiddleEncoder(nn.Module):
    """Mamba-based middle encoder that replaces DynamicVFE + SparseEncoder.

    Converts raw LiDAR points to BEV features via:
    1. Pillarization into BEV grid
    2. Conv1d per-pillar encoding
    3. Hilbert curve serialization of non-empty pillars
    4. Bidirectional Mamba blocks for long-range context
    5. Scatter back to dense BEV grid
    """

    def __init__(self,
                 in_channels=5,
                 output_channels=512,
                 point_cloud_range=None,
                 bev_size=180,
                 max_points_per_pillar=32,
                 trans_dim=384,
                 depth=12,
                 encoder_dims=384,
                 drop_path=0.1,
                 drop_out=0.0,
                 rms_norm=False):
        super().__init__()

        if point_cloud_range is None:
            point_cloud_range = [-54, -54, -5, 54, 54, 3]

        self.in_channels = in_channels
        self.output_channels = output_channels
        self.pc_range = point_cloud_range
        self.bev_h = bev_size
        self.bev_w = bev_size
        self.max_points_per_pillar = max_points_per_pillar
        self.trans_dim = trans_dim

        # Derived pillar size
        self.pillar_size_x = (point_cloud_range[3] - point_cloud_range[0]) / bev_size
        self.pillar_size_y = (point_cloud_range[4] - point_cloud_range[1]) / bev_size

        # Per-pillar point encoder (Conv1d + max-pool)
        self.pillar_encoder = PillarEncoder(in_channels, encoder_dims)

        # Positional embedding for pillar centers
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, trans_dim)
        )

        # Mamba blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = MixerModel(
            d_model=trans_dim,
            n_layer=depth,
            ssm_cfg=dict(use_fast_path=False),
            rms_norm=rms_norm,
            drop_out=drop_out,
            drop_path=dpr,
        )

        # OrderScale for bidirectional serialization
        self.OrderScale_gamma_1, self.OrderScale_beta_1 = init_OrderScale(trans_dim)
        self.OrderScale_gamma_2, self.OrderScale_beta_2 = init_OrderScale(trans_dim)

        # Project from trans_dim to output_channels
        self.proj_out = nn.Linear(trans_dim, output_channels)

        self._init_weights()

    def _init_weights(self):
        for m in [self.pillar_encoder, self.pos_embed, self.proj_out]:
            for p in m.modules():
                if isinstance(p, nn.Linear):
                    nn.init.xavier_uniform_(p.weight)
                    if p.bias is not None:
                        nn.init.zeros_(p.bias)
                elif isinstance(p, nn.Conv1d):
                    nn.init.xavier_uniform_(p.weight)
                    if p.bias is not None:
                        nn.init.zeros_(p.bias)

    def pillarize(self, points_list):
        """Bin raw points into BEV pillars.

        Args:
            points_list: list of (Ni, C) tensors, one per batch sample.

        Returns:
            pillar_features: (total_pillars, max_points_per_pillar, C) — padded
            pillar_centers: (total_pillars, 3) — XY center + mean Z
            pillar_coords: (total_pillars, 2) — (xi, yi) grid indices
            batch_indices: (total_pillars,) — batch index per pillar
            num_pillars_per_sample: list of int
        """
        all_pillar_feats = []
        all_pillar_centers = []
        all_pillar_coords = []
        all_batch_idx = []
        num_pillars_per_sample = []

        for b, points in enumerate(points_list):
            # points: (N, C) where C >= 3 (x, y, z, ...)
            x = points[:, 0]
            y = points[:, 1]

            # Compute grid indices
            xi = ((x - self.pc_range[0]) / self.pillar_size_x).long()
            yi = ((y - self.pc_range[1]) / self.pillar_size_y).long()

            # Clamp to valid range
            xi = xi.clamp(0, self.bev_w - 1)
            yi = yi.clamp(0, self.bev_h - 1)

            # Filter points outside range
            mask = (x >= self.pc_range[0]) & (x < self.pc_range[3]) & \
                   (y >= self.pc_range[1]) & (y < self.pc_range[4]) & \
                   (points[:, 2] >= self.pc_range[2]) & (points[:, 2] < self.pc_range[5])
            points = points[mask]
            xi = xi[mask]
            yi = yi[mask]

            if points.shape[0] == 0:
                # Handle empty point cloud
                num_pillars_per_sample.append(0)
                continue

            # Hash grid coordinates to unique pillar IDs
            flat_idx = yi * self.bev_w + xi  # (N,)

            # Get unique pillars and group points
            unique_ids, inverse = torch.unique(flat_idx, return_inverse=True)
            num_pillars = unique_ids.shape[0]

            # For each unique pillar, collect up to max_points_per_pillar points
            pillar_feats = torch.zeros(num_pillars, self.max_points_per_pillar,
                                       self.in_channels, device=points.device,
                                       dtype=points.dtype)
            pillar_counts = torch.zeros(num_pillars, device=points.device, dtype=torch.long)

            # Scatter points into pillars (truncate to max_points_per_pillar)
            for i in range(num_pillars):
                pillar_mask = (inverse == i)
                pillar_pts = points[pillar_mask]
                n_pts = min(pillar_pts.shape[0], self.max_points_per_pillar)
                # Take first n_pts (could also random sample)
                pillar_feats[i, :n_pts] = pillar_pts[:n_pts]
                pillar_counts[i] = n_pts

            # Compute pillar centers: grid center XY + mean Z
            pillar_yi = unique_ids // self.bev_w
            pillar_xi = unique_ids % self.bev_w
            center_x = (pillar_xi.float() + 0.5) * self.pillar_size_x + self.pc_range[0]
            center_y = (pillar_yi.float() + 0.5) * self.pillar_size_y + self.pc_range[1]

            # Mean Z from actual points in each pillar
            center_z = torch.zeros(num_pillars, device=points.device, dtype=points.dtype)
            for i in range(num_pillars):
                cnt = pillar_counts[i]
                if cnt > 0:
                    center_z[i] = pillar_feats[i, :cnt, 2].mean()

            centers = torch.stack([center_x, center_y, center_z], dim=1)  # (K, 3)
            coords = torch.stack([pillar_xi, pillar_yi], dim=1)  # (K, 2)

            all_pillar_feats.append(pillar_feats)
            all_pillar_centers.append(centers)
            all_pillar_coords.append(coords)
            all_batch_idx.append(torch.full((num_pillars,), b, device=points.device, dtype=torch.long))
            num_pillars_per_sample.append(num_pillars)

        if sum(num_pillars_per_sample) == 0:
            device = points_list[0].device
            bev = torch.zeros(len(points_list), self.output_channels,
                              self.bev_h, self.bev_w, device=device)
            return bev, None, None, None, num_pillars_per_sample

        pillar_features = torch.cat(all_pillar_feats, dim=0)
        pillar_centers = torch.cat(all_pillar_centers, dim=0)
        pillar_coords = torch.cat(all_pillar_coords, dim=0)
        batch_indices = torch.cat(all_batch_idx, dim=0)

        return pillar_features, pillar_centers, pillar_coords, batch_indices, num_pillars_per_sample

    def forward(self, raw_points, batch_size, **kwargs):
        """
        Args:
            raw_points: list of (Ni, C) tensors — raw points per sample
            batch_size: int
            **kwargs: passed through

        Returns:
            bev_features: (B, output_channels, bev_h, bev_w)
            encode_features: [] (empty, for interface compatibility)
            kwargs: passed through
        """
        device = raw_points[0].device

        # 1. Pillarization — rapid BEV grid generation
        pillar_features, pillar_centers, pillar_coords, batch_indices, num_pillars = \
            self.pillarize(raw_points)

        # Handle edge case: all empty
        if pillar_centers is None:
            bev = torch.zeros(batch_size, self.output_channels,
                              self.bev_h, self.bev_w, device=device)
            return bev, [], kwargs

        # 2. Conv1d Encoder — per-pillar feature extraction
        # pillar_features: (total_K, P, C) -> (total_K, encoder_dims)
        encoded = self.pillar_encoder(pillar_features)

        # 3. Positional embedding from pillar centers
        pos_encoded = self.pos_embed(pillar_centers)  # (total_K, trans_dim)

        # 4. Hilbert serialization per sample
        # We need to organize tokens into (B, K_max, trans_dim) batched tensors
        max_k = max(num_pillars)
        tokens_batched = torch.zeros(batch_size, max_k, self.trans_dim,
                                     device=device, dtype=encoded.dtype)
        pos_batched = torch.zeros(batch_size, max_k, self.trans_dim,
                                  device=device, dtype=pos_encoded.dtype)
        centers_batched = torch.zeros(batch_size, max_k, 3,
                                      device=device, dtype=pillar_centers.dtype)
        coords_batched = torch.zeros(batch_size, max_k, 2,
                                     device=device, dtype=torch.long)

        offset = 0
        for b in range(batch_size):
            k = num_pillars[b]
            if k > 0:
                tokens_batched[b, :k] = encoded[offset:offset + k]
                pos_batched[b, :k] = pos_encoded[offset:offset + k]
                centers_batched[b, :k] = pillar_centers[offset:offset + k]
                coords_batched[b, :k] = pillar_coords[offset:offset + k]
            offset += k

        # Radial/azimuthal serialization
        _, _, inv_fwd, tokens_forward, pos_forward = serialization_func(
            centers_batched, tokens_batched, pos_batched, 'radial')
        _, _, inv_bwd, tokens_backward, pos_backward = serialization_func(
            centers_batched, tokens_batched, pos_batched, 'azimuthal')

        # Apply OrderScale
        tokens_forward = apply_OrderScale(
            tokens_forward, self.OrderScale_gamma_1, self.OrderScale_beta_1)
        tokens_backward = apply_OrderScale(
            tokens_backward, self.OrderScale_gamma_2, self.OrderScale_beta_2)

        # Concatenate forward + backward: (B, 2*K_max, trans_dim)
        tokens_cat = torch.cat([tokens_forward, tokens_backward], dim=1)
        pos_cat = torch.cat([pos_forward, pos_backward], dim=1)

        # 5. Mamba blocks
        x = self.blocks(tokens_cat, pos_cat)  # (B, 2*K_max, trans_dim)

        # 6. Split forward/backward, unsort, and average
        x_fwd = x[:, :max_k, :]  # (B, K_max, trans_dim)
        x_bwd = x[:, max_k:, :]  # (B, K_max, trans_dim)

        # Unsort back to original pillar order using inverse indices
        x_fwd_unsorted = torch.zeros_like(x_fwd)
        x_bwd_unsorted = torch.zeros_like(x_bwd)

        # inv_fwd/inv_bwd shape: (1, B*K_max) from serialization_func
        # We need to handle the reshape carefully
        for b in range(batch_size):
            k = num_pillars[b]
            if k > 0:
                # The inverse order maps sorted position -> original position
                inv_f = inv_fwd[0, b * max_k: b * max_k + max_k]  # (K_max,)
                inv_b = inv_bwd[0, b * max_k: b * max_k + max_k]  # (K_max,)
                # Remap relative to this sample
                inv_f = inv_f - b * max_k
                inv_b = inv_b - b * max_k
                inv_f = inv_f.clamp(0, max_k - 1)
                inv_b = inv_b.clamp(0, max_k - 1)
                x_fwd_unsorted[b] = x_fwd[b][inv_f]
                x_bwd_unsorted[b] = x_bwd[b][inv_b]

        # Average the two directions
        x_avg = (x_fwd_unsorted + x_bwd_unsorted) / 2.0  # (B, K_max, trans_dim)

        # 7. Project to output channels and scatter to BEV
        x_proj = self.proj_out(x_avg)  # (B, K_max, output_channels)

        bev = torch.zeros(batch_size, self.output_channels, self.bev_h, self.bev_w,
                          device=device, dtype=x_proj.dtype)

        for b in range(batch_size):
            k = num_pillars[b]
            if k > 0:
                xi = coords_batched[b, :k, 0].long()  # (k,)
                yi = coords_batched[b, :k, 1].long()  # (k,)
                feats = x_proj[b, :k, :]  # (k, output_channels)
                # Scatter: place features at BEV positions
                # For overlapping pillars (shouldn't happen with unique coords), last write wins
                bev[b, :, yi, xi] = feats.T

        return bev, [], kwargs
