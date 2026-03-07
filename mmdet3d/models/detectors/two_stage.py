# Copyright (c) OpenMMLab. All rights reserved.
# ----------------- 物理修复：针对 mmdet.models 组件缺失 -----------------
try:
    from mmdet.models import DETECTORS, TwoStageDetector
except ImportError:
    # 适配 MMDet 3.x / MMEngine 路径
    try:
        from mmdet.models import DETECTORS
        from mmdet.models.detectors import TwoStageDetector
    except ImportError:
        # 兼容旧版 builder 路径
        from mmdet.models.builder import DETECTORS
        from mmdet.models.detectors.two_stage import TwoStageDetector
# ----------------- 物理修复结束 -----------------
from .base import Base3DDetector


@DETECTORS.register_module()
class TwoStage3DDetector(Base3DDetector, TwoStageDetector):
    """Base class of two-stage 3D detector.

    It inherits original ``:class:TwoStageDetector`` and
    ``:class:Base3DDetector``. This class could serve as a base class for all
    two-stage 3D detectors.
    """

    def __init__(self, **kwargs):
        super(TwoStage3DDetector, self).__init__(**kwargs)
