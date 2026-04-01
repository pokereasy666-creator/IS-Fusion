# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.runner import Runner


class CustomEpochBasedRunner(Runner):
    """Custom runner that calls set_epoch on datasets each epoch.

    .. deprecated::
        Prefer using :class:`EpochSyncHook` (registered via ``custom_hooks``
        in the config) together with the standard ``Runner``.  This class is
        kept only for backward compatibility.
    """

    def train(self):
        # Propagate current epoch to dataset pipeline transforms
        dataloader = self.train_dataloader
        if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, 'set_epoch'):
            dataloader.dataset.set_epoch(self.epoch)
        return super().train()
