# IS-Fusion: Method Overview

IS-Fusion is a multi-modal 3D object detection framework that fuses **LiDAR point clouds** and **multi-view camera images** for autonomous driving. It operates on the NuScenes dataset with 10 object classes.

The key innovation is the **Instance-Scene Fusion** mechanism, which bridges instance-level and scene-level representations through attention-based cross-modal interaction.

---

## Block Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              IS-Fusion: Method Overview                                 │
└─────────────────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────┐                          ┌──────────────────────┐
  │   Multi-view Images  │                          │  LiDAR Point Cloud   │
  │  (B, 6, 3, 384,1056) │                          │     (B, N, 5)        │
  └──────────┬───────────┘                          └──────────┬───────────┘
             │                                                  │
             ▼                                                  ▼
  ┌──────────────────────┐                          ┌──────────────────────┐
  │    Image Backbone    │                          │    Voxelization      │
  │ LaplacianSwinTransf. │                          │  voxel: 0.075x0.075 │
  │  (multi-scale feat.) │                          │    x0.2 meters      │
  └──────────┬───────────┘                          └──────────┬───────────┘
             │                                                  │
             │  [C192, C384, C768]                              ▼
             ▼                                      ┌──────────────────────┐
  ┌──────────────────────┐                          │  Voxel Encoder       │
  │    Image Neck        │                          │  DynamicVFE          │
  │  GeneralizedLSSFPN   │                          │  (N,5) -> (N,64)    │
  │  -> (B*6, 256, H, W) │                          └──────────┬───────────┘
  └──────────┬───────────┘                                      │
             │                                                  ▼
             │                                      ┌──────────────────────┐
             │                                      │  Middle Encoder      │
             │                                      │  SparseEncoder       │
             │                                      │  (3D Sparse Conv)    │
             │                                      │  -> (B,128,H_b,W_b) │
             │                                      └──────────┬───────────┘
             │                                                  │
             │                                                  │
             ▼                                                  ▼
  ┌───────────────────────────────────────────────────────────────────────┐
  │                                                                       │
  │                     ISFusionEncoder (Fusion Module)                    │
  │                                                                       │
  │  ┌─────────────────────────────────────────────────────────────────┐  │
  │  │  1. Image FV-to-BEV Projection (img_fv_to_bev)                 │  │
  │  │     - Create pillars from LiDAR points                         │  │
  │  │     - Project pillar 3D coords -> 2D image coords              │  │
  │  │       (using lidar2image + img_aug_matrix)                     │  │
  │  │     - Sample image features at projected locations              │  │
  │  │       (F.grid_sample across 6 cameras)                         │  │
  │  │     - Scatter sampled features onto BEV grid                   │  │
  │  │     -> img_bev_feats (B, 256, H_b, W_b)                       │  │
  │  └─────────────────────────┬───────────────────────────────────────┘  │
  │                            │                                          │
  │                            ▼                                          │
  │  ┌─────────────────────────────────────────────────────────────────┐  │
  │  │  2. BEV Feature Fusion (conv_fusion)                            │  │
  │  │     - Concatenate [img_bev_feats, lidar_feats]                  │  │
  │  │     - Conv2d: (256*3) -> (128)                                  │  │
  │  │     -> fused_bev (B, 128, H_b, W_b)                            │  │
  │  └─────────────────────────┬───────────────────────────────────────┘  │
  │                            │                                          │
  │                            ▼                                          │
  │  ┌─────────────────────────────────────────────────────────────────┐  │
  │  │  3. Region-based Sparse Attention (SST)                         │  │
  │  │     - SSTInputLayerV2: partition BEV into regions (6x6)         │  │
  │  │     - SSTv2: region-level self-attention (8 heads, 4 blocks)    │  │
  │  │     -> scene_feats                                              │  │
  │  └─────────────────────────┬───────────────────────────────────────┘  │
  │                            │                                          │
  │                            ▼                                          │
  │  ┌─────────────────────────────────────────────────────────────────┐  │
  │  │  4. Instance Fusion (instance_fusion)                           │  │
  │  │                                                                 │  │
  │  │  ┌───────────────────────────────────────────────────────────┐  │  │
  │  │  │ a) Instance Heatmap Prediction                            │  │  │
  │  │  │    - Conv layers -> per-class heatmap (B, 10, H_b, W_b)  │  │  │
  │  │  │    - Local NMS -> top-K instance proposals (K=200)        │  │  │
  │  │  └──────────────────────┬────────────────────────────────────┘  │  │
  │  │                         │                                       │  │
  │  │                         ▼                                       │  │
  │  │  ┌───────────────────────────────────────────────────────────┐  │  │
  │  │  │ b) Instance Context Attention (InsContextAtt)             │  │  │
  │  │  │    - Gather instance features at proposal locations       │  │  │
  │  │  │    - Self-attention among instances (2 layers)            │  │  │
  │  │  │    - Cross-attention: instances <-> scene features        │  │  │
  │  │  └──────────────────────┬────────────────────────────────────┘  │  │
  │  │                         │                                       │  │
  │  │                         ▼                                       │  │
  │  │  ┌───────────────────────────────────────────────────────────┐  │  │
  │  │  │ c) Instance-to-Scene Attention (Instane2SceneAtt)         │  │  │
  │  │  │    - Scatter refined instance features back to BEV        │  │  │
  │  │  │    - Cross-attention: scene queries <-> instance keys     │  │  │
  │  │  │    -> enhanced BEV features                               │  │  │
  │  │  └──────────────────────┬────────────────────────────────────┘  │  │
  │  │                         │                                       │  │
  │  └─────────────────────────┼───────────────────────────────────────┘  │
  │                            │                                          │
  │                            ▼                                          │
  │  ┌─────────────────────────────────────────────────────────────────┐  │
  │  │  5. Multi-stage Processing (pts_backbone stages)                │  │
  │  │     - Stage 1: region attention + instance fusion + 2D conv     │  │
  │  │     - Stage 2: region attention + 2D conv                       │  │
  │  │     -> multi-scale BEV features                                 │  │
  │  └─────────────────────────┬───────────────────────────────────────┘  │
  │                            │                                          │
  └────────────────────────────┼──────────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Points Backbone   │
                    │      SECONDV2        │
                    │  2D Conv on BEV      │
                    │  -> [C128, C256]      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Points Neck       │
                    │     SECONDFPN        │
                    │  Upsample + Concat   │
                    │  -> (B, 512, H, W)   │
                    └──────────┬───────────┘
                               │
                               ▼
  ┌───────────────────────────────────────────────────────────────────────┐
  │                    TransFusionHeadV2 (Detection Head)                 │
  │                                                                       │
  │  ┌─────────────────────────────────────────────────────────────────┐  │
  │  │  1. Heatmap-based Proposal Generation                           │  │
  │  │     - BEV features -> class-wise heatmap                        │  │
  │  │     - Select top-100 proposals                                  │  │
  │  └─────────────────────────┬───────────────────────────────────────┘  │
  │                            │                                          │
  │                            ▼                                          │
  │  ┌─────────────────────────────────────────────────────────────────┐  │
  │  │  2. Transformer Decoder                                         │  │
  │  │     - Self-attention: instance-to-instance                      │  │
  │  │     - Cross-attention: instances attend to BEV features         │  │
  │  │     - FFN refinement                                            │  │
  │  └─────────────────────────┬───────────────────────────────────────┘  │
  │                            │                                          │
  │                            ▼                                          │
  │  ┌─────────────────────────────────────────────────────────────────┐  │
  │  │  3. Prediction Heads                                            │  │
  │  │     - center (2D) | height (1D) | dim (3D) | rot (2D) | vel(2D)│  │
  │  └─────────────────────────┬───────────────────────────────────────┘  │
  │                            │                                          │
  └────────────────────────────┼──────────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Post-Processing    │
                    │  TransFusionBBoxCoder│
                    │  - Decode to world   │
                    │  - Circle NMS        │
                    │  - Score threshold   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    3D Detections     │
                    │  (x,y,z,w,l,h,yaw,  │
                    │   vx,vy,class,score) │
                    └──────────────────────┘
```

---

## Component Summary

| Component | Class | File |
|---|---|---|
| Main Detector | `ISFusionDetector` | `mmdet3d/models/detectors/isfusion.py` |
| Image Backbone | `LaplacianSwinTransformer` | `mmdet3d/models/backbones/laplacian_swin.py` |
| Image Neck | `GeneralizedLSSFPN` | `mmdet3d/models/necks/generalized_lss.py` |
| Voxel Encoder | `DynamicVFE` | `mmdet3d/models/voxel_encoders/voxel_fusion_encoder.py` |
| Middle Encoder | `SparseEncoder` | `mmdet3d/models/middle_encoders/sparse_encoder.py` |
| Fusion Encoder | `ISFusionEncoder` | `mmdet3d/models/middle_encoders/fusion_encoder.py` |
| Instance Attention | `InsContextAtt` | `mmdet3d/models/middle_encoders/fusion_encoder.py` |
| Instance-Scene Attention | `Instane2SceneAtt` | `mmdet3d/models/middle_encoders/fusion_encoder.py` |
| Points Backbone | `SECONDV2` | `mmdet3d/models/backbones/second.py` |
| Points Neck | `SECONDFPN` | `mmdet3d/models/necks/second_fpn.py` |
| Detection Head | `TransFusionHeadV2` | `mmdet3d/models/dense_heads/transfusion_head_v2.py` |
| BBox Coder | `TransFusionBBoxCoder` | `mmdet3d/core/bbox/coders/transfusion_bbox_coder.py` |

## Training Configuration

- **Dataset:** NuScenes (10 classes)
- **Voxel size:** 0.075 x 0.075 x 0.2 m
- **Point cloud range:** [-54, -54, -5] to [54, 54, 3] m
- **Optimizer:** AdamW (lr=6.25e-6, weight_decay=0.01)
- **Scheduler:** CosineAnnealingLR
- **Epochs:** 10
- **Matching:** HungarianAssigner3D

## Training Losses

```
Total Loss = FocalLoss(cls, gamma=2, alpha=0.25)
           + 0.25 * L1Loss(bbox regression)
           + 1.0  * GaussianFocalLoss(heatmap)
           + Instance heatmap auxiliary loss

Matching: HungarianAssigner3D (FocalLossCost + BBoxBEVL1Cost + IoU3DCost)
```
