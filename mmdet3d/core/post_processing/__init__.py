# Copyright (c) OpenMMLab. All rights reserved.
try:
    from mmdet.models.utils import (merge_aug_bboxes, merge_aug_masks,
                                   merge_aug_proposals, merge_aug_scores)
    from mmdet.utils import multiclass_nms
except ImportError:
    try:
        from mmdet.core.post_processing import (merge_aug_bboxes, merge_aug_masks,
                                                merge_aug_proposals,
                                                merge_aug_scores,
                                                multiclass_nms)
    except ImportError:
        def multiclass_nms(*args, **kwargs): pass
        def merge_aug_bboxes(*args, **kwargs): pass
        merge_aug_masks = merge_aug_bboxes
        merge_aug_proposals = merge_aug_bboxes
        merge_aug_scores = merge_aug_bboxes
from .box3d_nms import aligned_3d_nms, box3d_multiclass_nms, circle_nms
from .merge_augs import merge_aug_bboxes_3d

__all__ = [
    'multiclass_nms', 'merge_aug_proposals', 'merge_aug_bboxes',
    'merge_aug_scores', 'merge_aug_masks', 'box3d_multiclass_nms',
    'aligned_3d_nms', 'merge_aug_bboxes_3d', 'circle_nms'
]
