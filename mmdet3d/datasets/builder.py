# Copyright (c) OpenMMLab. All rights reserved.
import platform
from functools import partial

from mmdet3d.compat import Registry, build_from_cfg, collate

# ── Registries ──
try:
    from mmdet.registry import DATASETS
except ImportError:
    DATASETS = Registry('dataset')

try:
    from mmdet3d.datasets.dataset_wrappers import CBGSDataset
except ImportError:
    def CBGSDataset(dataset, *args, **kwargs):
        return dataset

if platform.system() != 'Windows':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

OBJECTSAMPLERS = Registry('Object sampler')
PIPELINES = Registry('pipeline')


def build_dataset(cfg, default_args=None):
    from torch.utils.data import ConcatDataset

    try:
        from mmdet.datasets.dataset_wrappers import RepeatDataset, ClassBalancedDataset
    except ImportError:
        RepeatDataset = None
        ClassBalancedDataset = None

    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']])
    elif cfg['type'] == 'RepeatDataset' and RepeatDataset is not None:
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset' and ClassBalancedDataset is not None:
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif cfg['type'] == 'CBGSDataset':
        dataset = CBGSDataset(build_dataset(cfg['dataset'], default_args))
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


def build_dataloader(dataset, samples_per_gpu, workers_per_gpu, num_gpus=1,
                     dist=False, shuffle=True, seed=None, persistent_workers=False,
                     **kwargs):
    """Build a PyTorch DataLoader."""
    from torch.utils.data import DataLoader

    sampler = None
    if dist:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=samples_per_gpu,
        num_workers=workers_per_gpu,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers and workers_per_gpu > 0,
    )
