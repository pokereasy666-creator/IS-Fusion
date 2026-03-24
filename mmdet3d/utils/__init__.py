# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.registry import Registry
from mmengine.logging import print_log
from .collect_env import collect_env
from .logger import get_root_logger


def register_all_modules(init_default_scope=True):
    """Register all modules in mmdet3d into the registries.

    This is called by mmengine's registry auto-discovery mechanism.
    Delegates to the top-level implementation in mmdet3d.__init__.
    """
    import mmdet3d
    mmdet3d.register_all_modules(init_default_scope)


__all__ = [
    'Registry', 'get_root_logger', 'collect_env', 'print_log',
    'register_all_modules'
]
