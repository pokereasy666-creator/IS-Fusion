# Copyright (c) OpenMMLab. All rights reserved.
import torch
# ----------------- 物理修复：针对 BaseModule 路径迁移 -----------------
try:
    from mmcv.runner import BaseModule
except ImportError:
    # 适配 MMEngine / MMCV 2.0+
    try:
        from mmcv.runner import BaseModule
    except ImportError:
        import torch.nn as nn
        BaseModule = nn.Module # 终极保底
# ----------------- 物理修复结束 -----------------

from mmdet3d import ops
# ----------------- 物理修复：针对 ROI_EXTRACTORS 注册表缺失 -----------------
try:
    from mmdet.models.builder import ROI_EXTRACTORS
except ImportError:
    # 适配 MMDet 3.x / MMEngine 注册表
    try:
        from mmdet.registry import MODELS as ROI_EXTRACTORS
    except ImportError:
        # 最后的保底手动创建
        from mmcv.utils import Registry
        ROI_EXTRACTORS = Registry('roi_extractor')
# ----------------- 物理修复结束 -----------------


@ROI_EXTRACTORS.register_module()
class Single3DRoIAwareExtractor(BaseModule):
    """Point-wise roi-aware Extractor.

    Extract Point-wise roi features.

    Args:
        roi_layer (dict): The config of roi layer.
    """

    def __init__(self, roi_layer=None, init_cfg=None):
        super(Single3DRoIAwareExtractor, self).__init__(init_cfg=init_cfg)
        self.roi_layer = self.build_roi_layers(roi_layer)

    def build_roi_layers(self, layer_cfg):
        """Build roi layers using `layer_cfg`"""
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        roi_layers = layer_cls(**cfg)
        return roi_layers

    def forward(self, feats, coordinate, batch_inds, rois):
        """Extract point-wise roi features.

        Args:
            feats (torch.FloatTensor): Point-wise features with
                shape (batch, npoints, channels) for pooling.
            coordinate (torch.FloatTensor): Coordinate of each point.
            batch_inds (torch.LongTensor): Indicate the batch of each point.
            rois (torch.FloatTensor): Roi boxes with batch indices.

        Returns:
            torch.FloatTensor: Pooled features
        """
        pooled_roi_feats = []
        for batch_idx in range(int(batch_inds.max()) + 1):
            roi_inds = (rois[..., 0].int() == batch_idx)
            coors_inds = (batch_inds.int() == batch_idx)
            pooled_roi_feat = self.roi_layer(rois[..., 1:][roi_inds],
                                             coordinate[coors_inds],
                                             feats[coors_inds])
            pooled_roi_feats.append(pooled_roi_feat)
        pooled_roi_feats = torch.cat(pooled_roi_feats, 0)
        return pooled_roi_feats
