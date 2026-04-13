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

    DCs may be nested inside lists/tuples (e.g. after
    ``MultiScaleFlipAug3D``).  The function descends recursively so that
    ``[DC(tensor)]`` from each sample is collated correctly.

    Non-DC values fall through to ``default_collate``.
    """
    import torch
    from torch.utils.data.dataloader import default_collate
    from mmdet3d.compat import DataContainer

    def _collate_field(samples):
        """Collate a list of per-sample values for a single field."""
        first = samples[0]

        if isinstance(first, DataContainer):
            if first.cpu_only:
                return [s._data for s in samples]
            elif first.stack:
                return torch.stack([s._data for s in samples], dim=0)
            else:
                return [s._data for s in samples]
        elif isinstance(first, (list, tuple)):
            # Recurse into lists/tuples (e.g. TTA wrappers from
            # MultiScaleFlipAug3D): transpose list-of-lists then collate
            # each position.
            collated = []
            for i in range(len(first)):
                inner_samples = [s[i] for s in samples]
                collated.append(_collate_field(inner_samples))
            return type(first)(collated)
        elif isinstance(first, dict):
            return {
                k: _collate_field([s[k] for s in samples])
                for k in first
            }
        else:
            try:
                return default_collate(samples)
            except (TypeError, RuntimeError):
                return samples

    if not isinstance(batch[0], dict):
        return default_collate(batch)

    result = {}
    for key in batch[0]:
        samples = [d[key] for d in batch]
        result[key] = _collate_field(samples)

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
