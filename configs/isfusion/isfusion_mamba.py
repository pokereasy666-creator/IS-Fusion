_base_ = './isfusion_0075voxel.py'

# Override the voxel encoder and middle encoder for Mamba-based pipeline.
# pts_voxel_layer is kept (needed for pillar creation in ISFusionEncoder).
# pts_voxel_encoder is removed (set to None) — raw points go directly to MambaMiddleEncoder.

voxel_size = [0.075, 0.075, 0.2]
point_cloud_range = [-54, -54, -5, 54, 54, 3]
out_size_factor = 8
voxel_shape = int((point_cloud_range[3] - point_cloud_range[0]) // voxel_size[0])
bev_size = voxel_shape // out_size_factor

model = dict(
    # Remove the voxel encoder — MambaMiddleEncoder handles raw points directly
    pts_voxel_encoder=None,

    # Replace SparseEncoder with MambaMiddleEncoder
    pts_middle_encoder=dict(
        _delete_=True,
        type='MambaMiddleEncoder',
        in_channels=5,
        output_channels=512,
        point_cloud_range=point_cloud_range,
        bev_size=bev_size,               # 180
        max_points_per_pillar=32,
        trans_dim=384,
        depth=12,                         # 12 Mamba blocks
        encoder_dims=384,
        drop_path=0.1,
        drop_out=0.0,
        rms_norm=False,
    ),
)
