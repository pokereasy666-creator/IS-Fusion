# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import torch
from mmengine.config import Config, DictAction
from mmengine.dist import get_dist_info, init_dist
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation')
    parser.add_argument('--result_dir', help='directory where results are saved')
    parser.add_argument('--bs', type=int, default=1, help='batch size')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show_bev', action='store_true', help='show bev results')
    parser.add_argument('--show_dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple workers')
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
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
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

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results) with the argument "--out", "--eval", "--format-only", '
         '"--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
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
        batch_size = args.bs
        cfg.test_dataloader = _build_compat_test_dataloader(cfg, batch_size)

    # Build test_evaluator from legacy config if not already present
    if not cfg.get('test_evaluator'):
        if cfg.get('val_evaluator'):
            cfg.test_evaluator = cfg.val_evaluator.copy()
        else:
            # Fallback: use dataset.evaluate() style
            cfg.test_evaluator = dict(type='NuScenesMetric')

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
