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
        def multiclass_nms(*args, **kwargs):
            raise RuntimeError(
                'multiclass_nms is not available. Install mmdet (v2 or v3).')

        def merge_aug_bboxes(*args, **kwargs):
            raise RuntimeError(
                'merge_aug_bboxes is not available. Install mmdet (v2 or v3).')

        def merge_aug_masks(*args, **kwargs):
            raise RuntimeError(
                'merge_aug_masks is not available. Install mmdet (v2 or v3).')

        def merge_aug_proposals(*args, **kwargs):
            raise RuntimeError(
                'merge_aug_proposals is not available. Install mmdet (v2 or v3).')

        def merge_aug_scores(*args, **kwargs):
            raise RuntimeError(
                'merge_aug_scores is not available. Install mmdet (v2 or v3).')
from .box3d_nms import aligned_3d_nms, box3d_multiclass_nms, circle_nms
from .merge_augs import merge_aug_bboxes_3d

__all__ = [
    'multiclass_nms', 'merge_aug_proposals', 'merge_aug_bboxes',
    'merge_aug_scores', 'merge_aug_masks', 'box3d_multiclass_nms',
    'aligned_3d_nms', 'merge_aug_bboxes_3d', 'circle_nms'
]
