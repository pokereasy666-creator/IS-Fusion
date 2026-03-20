# Copyright (c) OpenMMLab. All rights reserved.
import random
import warnings

import numpy as np
import torch
from mmengine.runner import Runner


def set_random_seed(seed, deterministic=False):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """Train a 3D detector or segmentor using MMEngine Runner.

    Args:
        model: The model to train (unused with Runner, kept for API compat).
        dataset: The training dataset (unused with Runner, kept for API compat).
        cfg: The config object. Must be in MMEngine v2 format with
            train_cfg, train_dataloader, optim_wrapper, param_scheduler, etc.
        distributed: Whether to use distributed training.
        validate: Whether to validate during training.
        timestamp: Timestamp string.
        meta: Meta information dict.
    """
    runner = Runner.from_cfg(cfg)
    runner.train()
