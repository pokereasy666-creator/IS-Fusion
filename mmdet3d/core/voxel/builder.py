# Copyright (c) OpenMMLab. All rights reserved.
from mmdet3d.compat import build_from_cfg

from . import voxel_generator


def build_voxel_generator(cfg, **kwargs):
    """Builder of voxel generator."""
    if isinstance(cfg, voxel_generator.VoxelGenerator):
        return cfg
    elif isinstance(cfg, dict):
        cfg = cfg.copy()
        cfg['default_args'] = kwargs
        obj_type = cfg.pop('type')
        cls = getattr(voxel_generator, obj_type)
        return cls(**{k: v for k, v in cfg.items() if k != 'default_args'}, **kwargs)
    else:
        raise TypeError('Invalid type {} for building a sampler'.format(
            type(cfg)))
