# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time

import torch
from mmengine.config import Config, DictAction
from mmengine.dist import init_dist
from mmengine.runner import Runner

from mmdet3d import register_all_modules
from mmdet3d.utils import collect_env, get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train a 3D detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--extra_tag', type=str, default=None,
                        help='extra tag for this experiment')
    parser.add_argument('--resume-from',
                        help='the checkpoint file to resume from')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--deterministic', action='store_true',
                        help='whether to set deterministic options for CUDNN')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='override some settings in the used config')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def _build_compat_dataloader(data_cfg, workers, batch_size, shuffle, test_mode=False):
    """Build a v2-style dataloader dict from legacy data.train/val/test config."""
    ds_cfg = data_cfg.copy()
    ds_cfg.pop('samples_per_gpu', None)
    if test_mode:
        ds_cfg['test_mode'] = True
    return dict(
        batch_size=batch_size,
        num_workers=workers,
        persistent_workers=True if workers > 0 else False,
        drop_last=False if test_mode else True,
        sampler=dict(type='DefaultSampler', shuffle=shuffle),
        dataset=ds_cfg)


def _migrate_legacy_data_config(cfg):
    """Convert legacy cfg.data to train_dataloader/val_dataloader/test_dataloader."""
    if not cfg.get('data'):
        return
    batch_size = cfg.data.get('samples_per_gpu', 1)
    workers = cfg.data.get('workers_per_gpu', 4)
    if not cfg.get('train_dataloader') and cfg.data.get('train'):
        cfg.train_dataloader = _build_compat_dataloader(
            cfg.data.train, workers, batch_size, shuffle=True)
    if not cfg.get('val_dataloader') and cfg.data.get('val'):
        cfg.val_dataloader = _build_compat_dataloader(
            cfg.data.val, workers, batch_size, shuffle=False, test_mode=True)
    if not cfg.get('test_dataloader') and cfg.data.get('test'):
        cfg.test_dataloader = _build_compat_dataloader(
            cfg.data.test, workers, batch_size, shuffle=False, test_mode=True)
    if not cfg.get('val_evaluator'):
        cfg.val_evaluator = dict(type='NuScenesMetric')
    if not cfg.get('test_evaluator'):
        cfg.test_evaluator = dict(type='NuScenesMetric')
    if not cfg.get('val_cfg'):
        cfg.val_cfg = dict()
    if not cfg.get('test_cfg'):
        cfg.test_cfg = dict()


def main():
    args = parse_args()

    # Register all mmdet3d modules (models, datasets, metrics, hooks, etc.)
    register_all_modules()

    cfg = Config.fromfile(args.config)

    # Handle custom_imports from config
    if cfg.get('custom_imports', None):
        from mmengine.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # Enable TF32 for Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.extra_tag is not None:
        cfg.work_dir = osp.join(cfg.work_dir, args.extra_tag)

    if args.resume_from is not None:
        cfg.resume = True
        cfg.load_from = args.resume_from

    # Init distributed env first
    if args.launcher == 'none':
        cfg.launcher = 'none'
    else:
        cfg.launcher = args.launcher
        init_dist(args.launcher)

    # Create work_dir
    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)

    # Dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    # Init logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # Log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    logger.info(f'Config:\n{cfg.pretty_text}')

    # Set random seeds
    if args.seed is not None:
        from mmdet3d.apis import set_random_seed
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
        cfg.seed = args.seed

    # Migrate legacy data config if present
    _migrate_legacy_data_config(cfg)

    # Build the runner from config and launch training
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
