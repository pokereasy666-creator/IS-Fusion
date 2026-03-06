# Copyright (c) OpenMMLab. All rights reserved.
# ----------------- 物理修复开始 -----------------
try:
    from mmcv.utils import Registry, build_from_cfg, print_log
except ImportError:
    from mmcv.utils import Registry, build_from_cfg
    from mmcv.utils import print_log
# ----------------- 物理修复结束 -----------------

from .collect_env import collect_env
from .logger import get_root_logger

__all__ = [
    'Registry', 'build_from_cfg', 'get_root_logger', 'collect_env', 'print_log'
]
