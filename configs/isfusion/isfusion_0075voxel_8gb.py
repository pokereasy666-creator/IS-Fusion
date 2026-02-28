_base_ = ['./isfusion_0075voxel.py']

# 仅覆盖特征通道和目标上限
model = dict(
    # 缩小图像特征提取(Swin)的输出通道
    img_neck=dict(out_channels=128),
    
    # 缩小3D体素编码的输出通道
    pts_voxel_encoder=dict(feat_channels=[32, 32]),
    
    # 砍半 SparseEncoder (3D骨干) 的通道数，极大地减少哈希表显存
    pts_middle_encoder=dict(
        base_channels=16,
        in_channels=32,
        encoder_channels=(
            (16, 16, 32),
            (32, 32, 64),
            (64, 64, 128),
            (128, 128),
        ),
        output_channels=128
    ),
    
    # 对齐 2D Backbone 的输入和输出通道
    pts_backbone=dict(
        in_channels=64, out_channels=[64, 128]
    ),
    
    # 对齐 Neck 的通道
    pts_neck=dict(
        in_channels=[64, 128],
        out_channels=[128, 128]
    ),
    
    # 对齐并缩小检测头
    pts_bbox_head=dict(
        in_channels=256,  # 128 + 128
        hidden_channel=64,
        num_proposals=100
    ),
    
    # 缩小融合层参数
    fusion_encoder=dict(embed_dims=128,
        instance_num=100
    )
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2
)
