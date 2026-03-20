# Copyright (c) OpenMMLab. All rights reserved.
from mmdet3d.compat import AssignResult

try:
    from mmdet.core.bbox import BaseAssigner, MaxIoUAssigner
except ImportError:
    from mmdet.models.task_modules.assigners import BaseAssigner, MaxIoUAssigner

from .hungarian_assigner import HungarianAssigner3D, HeuristicAssigner3D, HungarianAssigner3DV2, HungarianAssigner3DV3
__all__ = ['BaseAssigner', 'MaxIoUAssigner', 'AssignResult', 'HungarianAssigner3D', 'HeuristicAssigner', 'HungarianAssigner3D', 'HungarianAssigner3DV3']
