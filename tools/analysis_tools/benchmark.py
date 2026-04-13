# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time
import torch
from mmengine.config import Config
import torch.nn as nn
from mmengine.runner import load_checkpoint

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector
from tools.misc.fuse_conv_bn import fuse_module


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--samples', type=int, default=2000, help='samples to benchmark')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='interval of logging')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    # resolve test dataset config: v2 layout (test_dataloader.dataset) or
    # legacy v1 layout (data.test)
    if hasattr(cfg, 'test_dataloader'):
        test_data_cfg = cfg.test_dataloader.dataset
        # unwrap CBGSDataset wrapper if present
        if test_data_cfg.get('type', '') == 'CBGSDataset':
            test_data_cfg = test_data_cfg.dataset
        num_workers = cfg.test_dataloader.get('num_workers', 2)
    else:
        test_data_cfg = cfg.data.test
        num_workers = cfg.data.get('workers_per_gpu', 2)
    test_data_cfg.test_mode = True

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(test_data_cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=num_workers,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    # Only pass top-level test_cfg if the model config doesn't already have one,
    # otherwise build_detector asserts "test_cfg specified in both".
    test_cfg = cfg.get('test_cfg') if not cfg.model.get('test_cfg') else None
    model = build_detector(cfg.model, test_cfg=test_cfg)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_module(model)

    model = nn.DataParallel(model, device_ids=[0])

    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0

    # benchmark with several samples and take the average
    for i, data in enumerate(data_loader):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % args.log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f'Done image [{i + 1:<3}/ {args.samples}], '
                      f'fps: {fps:.1f} img / s')

        if (i + 1) == args.samples:
            pure_inf_time += elapsed
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f'Overall fps: {fps:.1f} img / s')
            break


if __name__ == '__main__':
    main()
