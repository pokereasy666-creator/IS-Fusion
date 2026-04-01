# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook
from mmengine.runner import Runner


class _EpochSyncInternalHook(Hook):
    """Internal hook that propagates epoch to dataset each epoch."""

    priority = 'ABOVE_NORMAL'

    def before_train_epoch(self, runner):
        dataloader = runner.train_dataloader
        if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, 'set_epoch'):
            dataloader.dataset.set_epoch(runner.epoch)


class CustomEpochBasedRunner(Runner):
    """Custom runner that calls set_epoch on datasets every epoch.

    It registers an internal hook so that ``set_epoch`` is invoked at the
    start of **every** training epoch, not just once at ``train()`` entry.

    .. deprecated::
        Prefer using :class:`EpochSyncHook` (registered via ``custom_hooks``
        in the config) together with the standard ``Runner``.  This class is
        kept only for backward compatibility.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_hook(_EpochSyncInternalHook(), priority='ABOVE_NORMAL')
