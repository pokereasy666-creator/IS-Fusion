# Copyright (c) OpenMMLab. All rights reserved.
import platform
from functools import partial

from mmdet3d.registry import DATASETS, TRANSFORMS, OBJECTSAMPLERS

if platform.system() != 'Windows':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

# Keep PIPELINES as alias for backward compat within mmdet3d
PIPELINES = TRANSFORMS


def build_dataset(cfg, default_args=None):
    from torch.utils.data import ConcatDataset
    from mmengine.dataset import RepeatDataset, ClassBalancedDataset

    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']])
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif cfg['type'] == 'CBGSDataset':
        from mmdet3d.datasets.dataset_wrappers import CBGSDataset
        dataset = CBGSDataset(build_dataset(cfg['dataset'], default_args))
    else:
        dataset = DATASETS.build(cfg, default_args=default_args)

    return dataset


def build_dataloader(dataset, samples_per_gpu, workers_per_gpu, num_gpus=1,
                     dist=False, shuffle=True, seed=None,
                     persistent_workers=False, **kwargs):
    """Build a PyTorch DataLoader."""
    from torch.utils.data import DataLoader
    from torch.utils.data.dataloader import default_collate

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
        collate_fn=default_collate,
        persistent_workers=persistent_workers and workers_per_gpu > 0,
    )
