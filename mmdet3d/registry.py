# Copyright (c) OpenMMLab. All rights reserved.
"""Central registry definitions for mmdet3d (OpenMMLab v2).

All registries are defined here. Model files should import registries from
this module rather than from mmdet or mmengine directly.
"""
from mmengine.registry import (
    Registry,
    RUNNERS as MMENGINE_RUNNERS,
    TRANSFORMS as MMENGINE_TRANSFORMS,
)
from mmdet.registry import (
    MODELS as MMDET_MODELS,
    TASK_UTILS,
    DATASETS as MMDET_DATASETS,
)
from mmseg.registry import MODELS as MMSEG_MODELS

# ── Local registries with parent chain ──
MODELS = Registry('models', parent=MMDET_MODELS)
DATASETS = Registry('datasets', parent=MMDET_DATASETS, scope='mmdet3d')
TRANSFORMS = Registry('transforms', parent=MMENGINE_TRANSFORMS, scope='mmdet3d')
OBJECTSAMPLERS = Registry('object_samplers')
VTRANSFORMS = Registry('vtransforms')
RUNNERS = Registry('runners', parent=MMENGINE_RUNNERS, scope='mmdet3d')

# ── Convenience aliases (unified MODELS registry in v2) ──
BACKBONES = MMDET_MODELS
DETECTORS = MMDET_MODELS
HEADS = MMDET_MODELS
LOSSES = MMDET_MODELS
NECKS = MMDET_MODELS
ROI_EXTRACTORS = MMDET_MODELS
SHARED_HEADS = MMDET_MODELS
SEGMENTORS = MMSEG_MODELS
VOXEL_ENCODERS = MODELS
MIDDLE_ENCODERS = MODELS
FUSION_LAYERS = MODELS

# ── Task-specific aliases ──
BBOX_CODERS = TASK_UTILS
BBOX_ASSIGNERS = TASK_UTILS
BBOX_SAMPLERS = TASK_UTILS
MATCH_COST = TASK_UTILS
