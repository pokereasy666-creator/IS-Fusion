# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.logging import MMLogger


def get_root_logger(log_file=None, log_level='INFO', name='mmdet3d'):
    """Get the root logger with mmengine."""
    return MMLogger.get_instance(name, log_file=log_file, log_level=log_level)
