# Copyright (c) OpenMMLab. All rights reserved.
import logging
# ----------------- 物理修复开始 -----------------
try:
    from mmcv.utils import get_logger
except ImportError:
    # MMEngine 2.x 日志系统迁移
    from mmengine.logging import MMLogger
    
    def get_logger(name, log_file=None, log_level='INFO', file_mode='w'):
        # 使用 MMEngine 的单例模式获取 logger
        return MMLogger.get_instance(name, log_file=log_file, log_level=log_level, file_mode=file_mode)
# ----------------- 物理修复结束 -----------------


def get_root_logger(log_file=None, log_level=logging.INFO, name='mmdet3d'):
    """Get root logger and add a keyword filter to it.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "mmdet3d".

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.
        name (str, optional): The name of the root logger, also used as a
            filter keyword. Defaults to 'mmdet3d'.

    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name=name, log_file=log_file, log_level=log_level)

    # add a logging filter
    logging_filter = logging.Filter(name)
    logging_filter.filter = lambda record: record.find(name) != -1

    return logger
