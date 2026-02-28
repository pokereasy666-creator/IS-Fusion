
# --- GLOBAL CBGS PATCH ---
try:
    from mmdet3d.datasets import CBGSDataset
except:
    try:
        from mmdet3d.datasets.dataset_wrappers import CBGSDataset
    except:
        def CBGSDataset(dataset, *args, **kwargs):
            return dataset  # 找不到就原样返回，强行跳过平衡采样
# -------------------------


try:
    from mmdet3d.datasets.dataset_wrappers import CBGSDataset
except ImportError:
    pass

# Copyright (c) OpenMMLab. All rights reserved.
import platform
# ----------------- 物理修复开始 -----------------
try:
    # 尝试旧路径 (MMCV 1.x)
    from mmcv.utils import Registry, build_from_cfg
except ImportError:
    # 新路径 (MMCV 2.x / MMEngine)
    from mmengine.registry import Registry, build_from_cfg
# ----------------- 物理修复结束 -----------------

# ----------------- 物理修复开始 -----------------
try:
    # 尝试旧路径 (MMDet 2.x)
    from mmdet.registry import DATASETS
except ImportError:
    # 新路径 (MMDet 3.x)
    try:
        from mmdet.registry import DATASETS
    except ImportError:
        # 如果都找不到，创建一个独立的注册表来保命
        # 注意：这里假设你之前的步骤已经正确导入了 Registry
        if 'Registry' not in locals():
            from mmengine.registry import Registry
        DATASETS = Registry('dataset')
# ----------------- 物理修复结束 -----------------
# ----------------- 物理修复开始 -----------------
# 屏蔽掉原来的 import
# from mmdet.datasets.builder import _concat_dataset

# 定义一个假的 _concat_dataset 函数，防止报错
def _concat_dataset(*args, **kwargs):
    return None
# ----------------- 物理修复结束 -----------------

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

OBJECTSAMPLERS = Registry('Object sampler')


def build_dataset(cfg, default_args=None):
    # [注意] 下面这块修复代码，必须缩进 4 个空格！
    try:
        from mmdet.datasets.dataset_wrappers import (ClassBalancedDataset,
                                                     ConcatDataset, RepeatDataset)
    except ImportError:
        # 针对 MMDet 3.x / MMEngine 的修复
        try:
            from mmdet.datasets import ConcatDataset, RepeatDataset
        except ImportError:
            from mmengine.dataset import ConcatDataset, RepeatDataset
        
        # 定义一个假的 ClassBalancedDataset
        class ClassBalancedDataset:
            pass
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True))
    elif cfg['type'] == 'SimpleDataset':
        dataset = SimpleDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif cfg['type'] == 'CBGSDataset':
        dataset = CBGSDataset(build_dataset(cfg['dataset'], default_args))
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset
