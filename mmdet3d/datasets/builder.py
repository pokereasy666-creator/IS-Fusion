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


def _collate_dc(batch):
    """Collate a batch of dicts whose values may be DataContainer.

    Respects the DC flags set by DefaultFormatBundle3D / Collect3D:

    * ``cpu_only=True`` (img_metas, gt_bboxes_3d as box objects):
      gathered as a plain list — never touched by ``default_collate``.
    * ``stack=True`` (img): stacked into a single batched tensor via
      ``torch.stack``.
    * ``stack=False`` (points, voxels, gt_labels_3d):
      gathered as a plain list of tensors — not stacked, because samples
      may have different sizes.

    Non-DC values fall through to ``default_collate``.
    """
    import torch
    from torch.utils.data.dataloader import default_collate
    from mmdet3d.compat import DataContainer

    if not isinstance(batch[0], dict):
        return default_collate(batch)

    result = {}
    for key in batch[0]:
        samples = [d[key] for d in batch]
        first = samples[0]

        if isinstance(first, DataContainer):
            if first.cpu_only:
                result[key] = [s._data for s in samples]
            elif first.stack:
                result[key] = torch.stack([s._data for s in samples], dim=0)
            else:
                result[key] = [s._data for s in samples]
        else:
            try:
                result[key] = default_collate(samples)
            except (TypeError, RuntimeError):
                result[key] = samples

    return result


def build_dataloader(dataset, samples_per_gpu, workers_per_gpu, num_gpus=1,
                     dist=False, shuffle=True, seed=None,
                     persistent_workers=False, **kwargs):
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
        collate_fn=_collate_dc,
        persistent_workers=persistent_workers and workers_per_gpu > 0,
    )
