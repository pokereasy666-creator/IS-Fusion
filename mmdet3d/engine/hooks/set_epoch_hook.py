# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class SetEpochHook(Hook):
    """Call ``dataset.set_epoch(epoch)`` before each training epoch.

    This propagates the current epoch number to the dataset and its
    pipeline transforms (e.g. ``ObjectSampleV2``, ``ModalMask3D``) so
    that epoch-dependent behaviour such as ``stop_epoch`` works correctly.
    """

    priority = 'NORMAL'

    def before_train_epoch(self, runner) -> None:
        dataset = runner.train_dataloader.dataset
        if hasattr(dataset, 'set_epoch'):
            dataset.set_epoch(runner.epoch)
