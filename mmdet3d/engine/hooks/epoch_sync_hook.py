# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class EpochSyncHook(Hook):
    """Propagates the current epoch number to dataset pipeline transforms.

    Many data augmentation transforms (ObjectSampleV2, ModalMask3D, GridMask)
    have a ``stop_epoch`` parameter and rely on ``set_epoch()`` being called
    each epoch.  This hook bridges the runner's epoch counter with those
    transforms by calling ``dataset.set_epoch(epoch)`` at the start of every
    training epoch.

    The call chain is:
        EpochSyncHook.before_train_epoch
        -> CBGSDataset.set_epoch
        -> NuScenesDataset.set_epoch  (custom_3d.py)
        -> pipeline transform.set_epoch  (for each transform that has it)
    """

    priority = 'ABOVE_NORMAL'

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        dataloader = runner.train_dataloader
        dataset = dataloader.dataset
        if hasattr(dataset, 'set_epoch'):
            dataset.set_epoch(epoch)
