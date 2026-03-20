"""
Centralized OpenMMLab v2 compatibility module.

All mmengine/mmcv v2 imports are routed through here so that individual files
don't need try-except blocks.  When running under mmcv 2.x + mmengine, the
canonical import paths are used directly.
"""

import os as _os

# ──────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────
from mmengine.registry import Registry

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
from mmengine.config import Config, DictAction

# ──────────────────────────────────────────────
# Base model classes
# ──────────────────────────────────────────────
from mmengine.model import BaseModule

try:
    from mmengine.model import ModuleList
except ImportError:
    from torch.nn import ModuleList

try:
    from mmengine.model import Sequential
except ImportError:
    from torch.nn import Sequential

# ──────────────────────────────────────────────
# Weight initialisation helpers
# ──────────────────────────────────────────────
try:
    from mmengine.model.weight_init import (
        kaiming_init,
        xavier_init,
        normal_init,
        constant_init,
        trunc_normal_init,
        bias_init_with_prob,
        uniform_init,
    )
except ImportError:
    from mmengine.model.weight_init import (
        kaiming_init,
        xavier_init,
        normal_init,
        constant_init,
    )

    def trunc_normal_init(module, mean=0.0, std=1.0, a=-2.0, b=2.0, bias=0.0):
        from torch.nn.init import trunc_normal_
        if hasattr(module, 'weight') and module.weight is not None:
            trunc_normal_(module.weight, mean=mean, std=std, a=a, b=b)
        if hasattr(module, 'bias') and module.bias is not None:
            import torch.nn as nn
            nn.init.constant_(module.bias, bias)

    def bias_init_with_prob(prior_prob):
        import math
        return float(-math.log((1 - prior_prob) / prior_prob))

    def uniform_init(module, a=0, b=1, bias=0):
        import torch.nn as nn
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.uniform_(module.weight, a, b)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)


# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
from mmengine.logging import print_log

try:
    from mmengine.logging import MMLogger
except ImportError:
    import logging as _logging
    MMLogger = _logging.getLogger


def get_root_logger(log_file=None, log_level='INFO', name='mmdet3d'):
    """Get the root logger with mmengine."""
    try:
        logger = MMLogger.get_instance(name, log_file=log_file, log_level=log_level)
    except Exception:
        import logging
        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        if log_file is not None:
            handler = logging.FileHandler(log_file)
            logger.addHandler(handler)
    return logger


# ──────────────────────────────────────────────
# File I/O
# ──────────────────────────────────────────────
try:
    from mmengine.fileio import load as fileio_load, dump as fileio_dump
except ImportError:
    import json as _json
    import pickle as _pickle

    def fileio_load(file, **kwargs):
        if str(file).endswith('.json'):
            with open(file) as f:
                return _json.load(f)
        with open(file, 'rb') as f:
            return _pickle.load(f)

    def fileio_dump(obj, file, **kwargs):
        if str(file).endswith('.json'):
            with open(file, 'w') as f:
                _json.dump(obj, f)
        else:
            with open(file, 'wb') as f:
                _pickle.dump(obj, f)


# ──────────────────────────────────────────────
# Runner utilities
# ──────────────────────────────────────────────
try:
    from mmengine.dist import get_dist_info, init_dist
except ImportError:
    from mmengine.runner import get_dist_info, init_dist

try:
    from mmengine.runner import load_checkpoint
except ImportError:
    def load_checkpoint(model, filename, map_location=None, strict=False, logger=None):
        import torch
        checkpoint = torch.load(filename, map_location=map_location)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=strict)
        return checkpoint


# ──────────────────────────────────────────────
# FP16 decorators  (no-op in v2; native AMP is used instead)
# ──────────────────────────────────────────────
def force_fp32(apply_to=None, out_fp16=False):
    """No-op decorator -- kept for source compatibility with v1 code."""
    def decorator(func):
        return func
    return decorator


def auto_fp16(apply_to=None, out_fp32=False):
    """No-op decorator -- kept for source compatibility with v1 code."""
    def decorator(func):
        return func
    return decorator


# ──────────────────────────────────────────────
# DataContainer (removed in mmcv 2.x)
# ──────────────────────────────────────────────
try:
    from mmcv.parallel import DataContainer
except ImportError:
    class DataContainer:
        """Lightweight replacement for mmcv.parallel.DataContainer."""
        def __init__(self, data, cpu_only=False, stack=False,
                     pad_dims=None, padding_value=0):
            self._data = data
            self.cpu_only = cpu_only
            self.stack = stack
            self.pad_dims = pad_dims
            self.padding_value = padding_value

        @property
        def data(self):
            return self._data

        def __repr__(self):
            return f'{self.__class__.__name__}({self._data})'


# ──────────────────────────────────────────────
# Parallel wrappers (removed in mmcv 2.x)
# ──────────────────────────────────────────────
try:
    from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
except ImportError:
    import torch.nn as _nn
    MMDataParallel = _nn.DataParallel
    MMDistributedDataParallel = _nn.parallel.DistributedDataParallel

try:
    from mmcv.parallel import collate
except ImportError:
    def collate(batch, samples_per_gpu=1):
        """Simple collate fallback when mmcv.parallel.collate is unavailable."""
        import torch
        from torch.utils.data.dataloader import default_collate
        if isinstance(batch[0], DataContainer):
            return DataContainer([s.data for s in batch])
        return default_collate(batch)


# ──────────────────────────────────────────────
# Registry aliases (mmdet 3.x unified MODELS registry)
# ──────────────────────────────────────────────
try:
    from mmdet.registry import MODELS as _MMDET_MODELS
except ImportError:
    try:
        from mmengine.registry import MODELS as _MMDET_MODELS
    except ImportError:
        _MMDET_MODELS = Registry('models')

BACKBONES = _MMDET_MODELS
DETECTORS = _MMDET_MODELS
HEADS = _MMDET_MODELS
LOSSES = _MMDET_MODELS
NECKS = _MMDET_MODELS
ROI_EXTRACTORS = _MMDET_MODELS
SHARED_HEADS = _MMDET_MODELS

try:
    from mmdet.registry import TASK_UTILS as _TASK_UTILS
except ImportError:
    _TASK_UTILS = Registry('task_utils')

BBOX_CODERS = _TASK_UTILS
BBOX_ASSIGNERS = _TASK_UTILS
BBOX_SAMPLERS = _TASK_UTILS
MATCH_COST = _TASK_UTILS


# ──────────────────────────────────────────────
# Transformer registries (removed in mmcv 2.x, map to MODELS)
# ──────────────────────────────────────────────
try:
    from mmcv.cnn.bricks.registry import (
        ATTENTION,
        FEEDFORWARD_NETWORK,
        TRANSFORMER_LAYER,
        TRANSFORMER_LAYER_SEQUENCE,
    )
except (ImportError, ModuleNotFoundError):
    ATTENTION = _MMDET_MODELS
    FEEDFORWARD_NETWORK = _MMDET_MODELS
    TRANSFORMER_LAYER = _MMDET_MODELS
    TRANSFORMER_LAYER_SEQUENCE = _MMDET_MODELS


# ──────────────────────────────────────────────
# mmdet.core utilities (moved in mmdet 3.x)
# ──────────────────────────────────────────────
try:
    from mmdet.models.utils import multi_apply
except ImportError:
    def multi_apply(func, *args, **kwargs):
        pfunc = functools.partial(func, **kwargs) if kwargs else func
        map_results = map(pfunc, *args)
        return tuple(map(list, zip(*map_results)))

try:
    from mmdet.models.task_modules.builder import (
        build_bbox_coder,
        build_assigner,
        build_sampler,
    )
except ImportError:
    def build_bbox_coder(cfg):
        return _TASK_UTILS.build(cfg)
    def build_assigner(cfg):
        return _TASK_UTILS.build(cfg)
    def build_sampler(cfg, **default_args):
        return _TASK_UTILS.build(cfg, default_args=default_args)

try:
    from mmdet.models.task_modules.prior_generators import build_anchor_generator
except ImportError:
    def build_anchor_generator(cfg):
        return _TASK_UTILS.build(cfg)

try:
    from mmdet.models.task_modules.assigners import AssignResult
except ImportError:
    AssignResult = None

try:
    from mmdet.models.task_modules.samplers import PseudoSampler
except ImportError:
    class PseudoSampler:
        def __init__(self, **kwargs):
            pass
        def sample(self, assign_result, bboxes, gt_bboxes, **kwargs):
            import torch
            pos_inds = torch.nonzero(
                assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
            neg_inds = torch.nonzero(
                assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
            from collections import namedtuple
            SamplingResult = namedtuple(
                'SamplingResult', ['pos_inds', 'neg_inds', 'bboxes', 'gt_bboxes',
                                   'pos_bboxes', 'neg_bboxes', 'pos_gt_bboxes',
                                   'num_gts', 'pos_assigned_gt_inds',
                                   'pos_is_gt'])
            return SamplingResult(
                pos_inds=pos_inds, neg_inds=neg_inds,
                bboxes=bboxes, gt_bboxes=gt_bboxes,
                pos_bboxes=bboxes[pos_inds],
                neg_bboxes=bboxes[neg_inds],
                pos_gt_bboxes=gt_bboxes[assign_result.gt_inds[pos_inds] - 1]
                    if len(pos_inds) > 0 else gt_bboxes.new_empty((0, gt_bboxes.size(-1))),
                num_gts=gt_bboxes.shape[0],
                pos_assigned_gt_inds=assign_result.gt_inds[pos_inds] - 1
                    if len(pos_inds) > 0 else bboxes.new_empty(0, dtype=torch.long),
                pos_is_gt=bboxes.new_zeros(len(pos_inds), dtype=torch.uint8),
            )

try:
    from mmdet.structures.bbox import bbox_overlaps
except ImportError:
    bbox_overlaps = None

try:
    from mmdet.models.utils import images_to_levels
except ImportError:
    def images_to_levels(target, num_levels):
        """Convert targets by image to targets by feature level."""
        level_targets = []
        start = 0
        for n in num_levels:
            end = start + n
            level_targets.append(target[:, start:end].contiguous())
            start = end
        return level_targets

try:
    from mmdet.structures.bbox import bbox2result
except ImportError:
    def bbox2result(bboxes, labels, num_classes):
        import torch
        if bboxes.shape[0] == 0:
            return [bboxes.new_zeros(0, 5) for _ in range(num_classes)]
        return [bboxes[labels == i, :] for i in range(num_classes)]


# ──────────────────────────────────────────────
# Runner/optimizer building (v1 compat)
# ──────────────────────────────────────────────
try:
    from mmengine.runner import Runner
except ImportError:
    Runner = None

try:
    from mmengine.optim import build_optim_wrapper
except ImportError:
    build_optim_wrapper = None


def build_optimizer(model, cfg):
    """Build optimizer from config dict (v1-style interface)."""
    import torch.optim as optim
    optimizer_cfg = cfg.copy()
    optimizer_type = optimizer_cfg.pop('type')
    optimizer_cls = getattr(optim, optimizer_type)
    return optimizer_cls(model.parameters(), **optimizer_cfg)


class _StubHooksRegistry:
    """Stub for HOOKS registry when mmcv.runner is unavailable."""
    def register_module(self, *args, **kwargs):
        def wrapper(cls):
            return cls
        if args and callable(args[0]):
            return args[0]
        return wrapper

try:
    from mmcv.runner import HOOKS
except (ImportError, ModuleNotFoundError):
    HOOKS = _StubHooksRegistry()


# ──────────────────────────────────────────────
# Misc utilities that were in mmcv.utils
# ──────────────────────────────────────────────
try:
    from mmengine.utils import to_2tuple
except ImportError:
    def to_2tuple(x):
        if isinstance(x, (list, tuple)):
            return tuple(x)
        return (x, x)

try:
    from mmengine.utils import is_tuple_of, is_list_of
except ImportError:
    def is_tuple_of(seq, expected_type):
        return isinstance(seq, tuple) and all(isinstance(s, expected_type) for s in seq)

    def is_list_of(seq, expected_type):
        return isinstance(seq, list) and all(isinstance(s, expected_type) for s in seq)

try:
    from mmengine.utils import ProgressBar, track_iter_progress
except ImportError:
    def track_iter_progress(iterable, **kwargs):
        yield from iterable
    ProgressBar = None

try:
    from mmengine.utils.misc import import_modules_from_strings
except ImportError:
    def import_modules_from_strings(imports, allow_failed_imports=False):
        import importlib
        if isinstance(imports, str):
            imports = [imports]
        if imports is None:
            return
        for mod in imports:
            try:
                importlib.import_module(mod)
            except ImportError:
                if not allow_failed_imports:
                    raise


# ──────────────────────────────────────────────
# Convenience: mkdir_or_exist
# ──────────────────────────────────────────────
def mkdir_or_exist(dir_name, mode=0o777):
    _os.makedirs(dir_name, exist_ok=True)


# ──────────────────────────────────────────────
# build_from_cfg  (legacy helper)
# ──────────────────────────────────────────────
def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict using registry.build()."""
    return registry.build(cfg, default_args=default_args)


# ──────────────────────────────────────────────
# PyTorch 2.x checkpoint use_reentrant patch
# ──────────────────────────────────────────────
def patch_checkpoint_reentrant():
    """Patch torch.utils.checkpoint.checkpoint for PyTorch 2.x compat."""
    import torch
    import inspect
    _orig_cp = torch.utils.checkpoint.checkpoint
    if 'use_reentrant' in inspect.signature(_orig_cp).parameters:
        def _patched_cp(*args, **kwargs):
            kwargs.setdefault('use_reentrant', False)
            return _orig_cp(*args, **kwargs)
        torch.utils.checkpoint.checkpoint = _patched_cp


# ──────────────────────────────────────────────
# yapf compatibility patch
# ──────────────────────────────────────────────
def patch_yapf():
    """Remove unsupported 'verify' kwarg from yapf FormatCode."""
    try:
        import yapf.yapflib.yapf_api as _yapf_api
        _orig_FormatCode = _yapf_api.FormatCode
        def _patched_FormatCode(text, **kwargs):
            kwargs.pop('verify', None)
            return _orig_FormatCode(text, **kwargs)
        _yapf_api.FormatCode = _patched_FormatCode
    except Exception:
        pass


# ──────────────────────────────────────────────
# Ensure mmcv has expected top-level helpers
# ──────────────────────────────────────────────
def patch_mmcv():
    """Add missing top-level helpers to mmcv for v2 compat."""
    import mmcv
    if not hasattr(mmcv, 'load'):
        try:
            from mmengine.fileio import load
            mmcv.load = load
        except ImportError:
            mmcv.load = fileio_load
    if not hasattr(mmcv, 'dump'):
        try:
            from mmengine.fileio import dump
            mmcv.dump = dump
        except ImportError:
            mmcv.dump = fileio_dump
    if not hasattr(mmcv, 'mkdir_or_exist'):
        mmcv.mkdir_or_exist = mkdir_or_exist
    if not hasattr(mmcv, 'build_from_cfg'):
        mmcv.build_from_cfg = build_from_cfg
