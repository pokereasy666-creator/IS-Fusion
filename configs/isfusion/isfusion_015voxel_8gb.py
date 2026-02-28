_base_ = ['./isfusion_0075voxel.py']

# 统一放大 Voxel Size，降低分辨率，拯救 8GB 显存
voxel_size = [0.15, 0.15, 0.4]

model = dict(
    pts_voxel_layer=dict(
        max_voxels=(16000, 20000),
        voxel_size=voxel_size
    ),
    pts_voxel_encoder=dict(
        voxel_size=voxel_size
    ),
    pts_middle_encoder=dict(
        sparse_shape=[21, 720, 720]  # 41->21 (z轴), 1440->720 (x,y轴)
    ),
    pts_bbox_head=dict(
        num_proposals=100,
        bbox_coder=dict(
            voxel_size=[0.15, 0.15]  # 2D BBox coder 只需要 x, y
        )
    ),
    fusion_encoder=dict(
        instance_num=100,
        # 因为 x,y 分辨率减半(1440->720)，下采样8倍后的 bev_size 也减半(180->90)
        bev_size=90,  
        grid_size=[
            [90, 90, 1],  # level 0
            [45, 45, 1],  # level 1
        ]
    )
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2
)
