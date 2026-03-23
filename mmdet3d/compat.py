# Copyright (c) OpenMMLab. All rights reserved.
"""OpenMMLab v2 imports — clean, no backward compatibility shims.

This module re-exports commonly used symbols from mmengine / mmdet / mmseg
so that mmdet3d code has a single import source for shared utilities.
"""

# ── Registry ──
from mmengine.registry import Registry

# ── Config ──
from mmengine.config import Config, DictAction

# ── Base model classes ──
from mmengine.model import BaseModule, ModuleList, Sequential

# ── Weight initialisation helpers ──
from mmengine.model.weight_init import (
    kaiming_init,
    xavier_init,
    normal_init,
    constant_init,
    trunc_normal_init,
    bias_init_with_prob,
    uniform_init,
)

# ── Logging ──
from mmengine.logging import print_log, MMLogger


def get_root_logger(log_file=None, log_level='INFO', name='mmdet3d'):
    """Get the root logger with mmengine."""
    return MMLogger.get_instance(name, log_file=log_file, log_level=log_level)


# ── File I/O ──
from mmengine.fileio import load as fileio_load, dump as fileio_dump

# ── Distributed utilities ──
from mmengine.dist import get_dist_info, init_dist

# ── Runner ──
from mmengine.runner import Runner, load_checkpoint

# ── Optimizer ──
from mmengine.optim import build_optim_wrapper

# ── Misc utilities ──
from mmengine.utils import to_2tuple, is_tuple_of, is_list_of
from mmengine.utils import ProgressBar, track_iter_progress
from mmengine.utils.misc import import_modules_from_strings

# ── mmdet utilities (moved in v3) ──
from mmdet.models.utils import multi_apply
from mmdet.models.task_modules.builder import (
    build_bbox_coder,
    build_assigner,
    build_sampler,
)
try:
    from mmdet.models.task_modules.prior_generators import build_anchor_generator
except ImportError:
    from mmengine.registry import build_from_cfg
    from mmdet.registry import TASK_UTILS as _TASK_UTILS

    def build_anchor_generator(cfg, default_args=None):
        return build_from_cfg(cfg, _TASK_UTILS, default_args)
from mmdet.models.task_modules.assigners import AssignResult
from mmdet.models.task_modules.samplers import PseudoSampler
from mmdet.structures.bbox import bbox_overlaps

# ── Convenience re-exports from mmdet3d.registry ──
from mmdet3d.registry import (
    MODELS,
    BACKBONES,
    DETECTORS,
    HEADS,
    LOSSES,
    NECKS,
    ROI_EXTRACTORS,
    SHARED_HEADS,
    SEGMENTORS,
    VOXEL_ENCODERS,
    MIDDLE_ENCODERS,
    FUSION_LAYERS,
    TRANSFORMS,
    OBJECTSAMPLERS,
    VTRANSFORMS,
    BBOX_CODERS,
    BBOX_ASSIGNERS,
    BBOX_SAMPLERS,
    MATCH_COST,
    TASK_UTILS,
    DATASETS,
)


def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays."""
    import torch
    if bboxes.shape[0] == 0:
        return [bboxes.new_zeros(0, 5) for _ in range(num_classes)]
    return [bboxes[labels == i, :] for i in range(num_classes)]


def images_to_levels(target, num_levels):
    """Convert targets by image to targets by feature level."""
    level_targets = []
    start = 0
    for n in num_levels:
        end = start + n
        level_targets.append(target[:, start:end].contiguous())
        start = end
    return level_targets
