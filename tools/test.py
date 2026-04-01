# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import torch
from mmengine.config import Config, DictAction
from mmengine.dist import get_dist_info, init_dist
from mmengine.runner import Runner

from mmdet3d import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--bs', type=int, default=1, help='batch size')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def _build_compat_test_dataloader(cfg, batch_size):
    """Build a v2-style test_dataloader dict from legacy data.test config."""
    test_data_cfg = cfg.data.test.copy()
    workers = cfg.data.get('workers_per_gpu', 4)
    test_data_cfg.pop('samples_per_gpu', None)
    test_data_cfg['test_mode'] = True

    return dict(
        batch_size=batch_size,
        num_workers=workers,
        persistent_workers=True if workers > 0 else False,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=test_data_cfg)


def main():
    args = parse_args()

    # Register all mmdet3d modules (models, datasets, metrics, hooks, etc.)
    register_all_modules()

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)

    # Handle custom_imports from config
    if cfg.get('custom_imports', None):
        from mmengine.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # Set up launcher
    cfg.launcher = args.launcher
    if args.launcher != 'none':
        init_dist(args.launcher)

    # Load checkpoint
    cfg.load_from = args.checkpoint

    # Set work_dir for test outputs
    if cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # Build test_dataloader from legacy data.test if not already present
    if not cfg.get('test_dataloader'):
        if cfg.get('data') and cfg.data.get('test'):
            batch_size = args.bs
            cfg.test_dataloader = _build_compat_test_dataloader(cfg, batch_size)
        else:
            raise ValueError(
                'Config has no test_dataloader and no legacy data.test. '
                'Please define test_dataloader in your config.')

    # Build test_evaluator from legacy config if not already present
    if not cfg.get('test_evaluator'):
        _DATASET_TO_METRIC = {
            'NuScenesDataset': 'NuScenesMetric',
            'KittiDataset': 'KittiMetric',
            'WaymoDataset': 'WaymoMetric',
            'LyftDataset': 'LyftMetric',
            'SUNRGBDDataset': 'IndoorMetric',
            'ScanNetDataset': 'IndoorMetric',
            'S3DISDataset': 'IndoorMetric',
            'ScanNetSegDataset': 'SegMetric',
            'S3DISSegDataset': 'SegMetric',
            'SemanticKITTIDataset': 'SegMetric',
        }
        if cfg.get('val_evaluator'):
            cfg.test_evaluator = cfg.val_evaluator.copy()
        else:
            # Try to infer from legacy cfg.data.test or v2 test_dataloader
            ds_cfg = None
            if cfg.get('data') and cfg.data.get('test'):
                ds_cfg = cfg.data.test.copy()
            elif cfg.get('test_dataloader') and cfg.test_dataloader.get('dataset'):
                ds_cfg = cfg.test_dataloader.dataset.copy()

            if ds_cfg is not None:
                while ds_cfg.get('dataset'):
                    ds_cfg = ds_cfg['dataset']
                ds_type = ds_cfg.get('type', '')
                metric_type = _DATASET_TO_METRIC.get(ds_type)
                if metric_type is not None:
                    cfg.test_evaluator = dict(type=metric_type)
                else:
                    raise ValueError(
                        f'No evaluator metric registered for dataset type '
                        f'"{ds_type}". Please set test_evaluator explicitly '
                        f'in your config.')
            else:
                raise ValueError(
                    'Config has no test_evaluator and no dataset config to '
                    'infer from. Please define test_evaluator in your config.')

    # Ensure test_cfg exists
    if not cfg.get('test_cfg'):
        cfg.test_cfg = dict()

    # Set random seeds
    if args.seed is not None:
        from mmdet3d.apis import set_random_seed
        set_random_seed(args.seed, deterministic=args.deterministic)

    # Build runner and run test
    runner = Runner.from_cfg(cfg)
    metrics = runner.test()

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            from mmengine.fileio import dump
            dump(metrics, args.out)
        print(metrics)


if __name__ == '__main__':
    main()
