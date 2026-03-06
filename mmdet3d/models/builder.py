# Copyright (c) OpenMMLab. All rights reserved.
import warnings

# ----------------- 物理修复开始：适配 MMCV / MMEngine -----------------
try:
    # 尝试旧版路径 (MMCV 1.x)
    from mmcv.utils import Registry
    from mmcv.cnn import MODELS as MMCV_MODELS
except ImportError:
    # 新版路径 (MMEngine / MMCV 2.x)
    from mmcv.utils import Registry
    from mmcv.cnn import MODELS as MMCV_MODELS
# ----------------- 物理修复结束 -----------------

# ----------------- 物理修复开始：适配 MMDetection 3.x -----------------
try:
    # 尝试旧版路径 (MMDet 2.x)
    from mmdet.models.builder import (BACKBONES, DETECTORS, HEADS, LOSSES, NECKS,
                                      ROI_EXTRACTORS, SHARED_HEADS)
    from mmseg.models.builder import SEGMENTORS
except ImportError:
    # 新版路径 (MMDet 3.x / MMSeg 1.x)
    # 在 3.x 架构中，所有模块统一注册在 MODELS 里
    try:
        from mmdet.registry import MODELS
    except ImportError:
        # 最后的保底，直接用 MMCV 的
        MODELS = MMCV_MODELS

    # 建立别名引用，防止后续函数报错
    BACKBONES = MODELS
    DETECTORS = MODELS
    HEADS = MODELS
    LOSSES = MODELS
    NECKS = MODELS
    ROI_EXTRACTORS = MODELS
    SHARED_HEADS = MODELS
    
    try:
        from mmseg.registry import MODELS as SEG_MODELS
        SEGMENTORS = SEG_MODELS
    except ImportError:
        # 如果没装 mmseg，用通用模型库代替
        SEGMENTORS = MODELS
# ----------------- 物理修复结束 -----------------

MODELS = Registry('models', parent=MMCV_MODELS)
VTRANSFORMS = Registry("vtransforms")

VOXEL_ENCODERS = MODELS
MIDDLE_ENCODERS = MODELS
FUSION_LAYERS = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_vtransform(cfg):
    return VTRANSFORMS.build(cfg)


def build_roi_extractor(cfg):
    """Build RoI feature extractor."""
    return ROI_EXTRACTORS.build(cfg)


def build_shared_head(cfg):
    """Build shared head of detector."""
    return SHARED_HEADS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss function."""
    return LOSSES.build(cfg)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return DETECTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    """Build segmentor."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return SEGMENTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_model(cfg, train_cfg=None, test_cfg=None):
    """A function warpper for building 3D detector or segmentor according to
    cfg.

    Should be deprecated in the future.
    """
    if cfg.type in ['EncoderDecoder3D']:
        return build_segmentor(cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    else:
        return build_detector(cfg, train_cfg=train_cfg, test_cfg=test_cfg)


def build_voxel_encoder(cfg):
    """Build voxel encoder."""
    return VOXEL_ENCODERS.build(cfg)


def build_middle_encoder(cfg):
    """Build middle level encoder."""
    return MIDDLE_ENCODERS.build(cfg)


def build_fusion_layer(cfg):
    """Build fusion layer."""
    return FUSION_LAYERS.build(cfg)
