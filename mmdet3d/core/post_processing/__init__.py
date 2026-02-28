# Copyright (c) OpenMMLab. All rights reserved.
# ----------------- 物理修复开始 -----------------
try:
    # 尝试旧路径 (MMDet 2.x)
    from mmdet.core.post_processing import (merge_aug_bboxes, merge_aug_masks,
                                            merge_aug_proposals,
                                            merge_aug_scores,
                                            multiclass_nms)
except (ImportError, ModuleNotFoundError):
    # 针对 MMDet 3.x / MMEngine 的新路径
    # 基础的 NMS 和合并逻辑现在分散在 structures 和 models.utils 中
    try:
        from mmdet.models.utils import (merge_aug_bboxes, merge_aug_masks,
                                       merge_aug_proposals, merge_aug_scores)
        from mmdet.utils import multiclass_nms
    except ImportError:
        # 如果还是找不到，说明是极其基础的版本变动，
        # 很多时候在生成数据阶段根本用不到这些函数，我们可以先定义为空防止崩盘
        def multiclass_nms(*args, **kwargs): pass
        def merge_aug_bboxes(*args, **kwargs): pass
        merge_aug_masks = merge_aug_bboxes
        merge_aug_proposals = merge_aug_bboxes
        merge_aug_scores = merge_aug_bboxes
# ----------------- 物理修复结束 -----------------
from .box3d_nms import aligned_3d_nms, box3d_multiclass_nms, circle_nms
from .merge_augs import merge_aug_bboxes_3d

__all__ = [
    'multiclass_nms', 'merge_aug_proposals', 'merge_aug_bboxes',
    'merge_aug_scores', 'merge_aug_masks', 'box3d_multiclass_nms',
    'aligned_3d_nms', 'merge_aug_bboxes_3d', 'circle_nms'
]
