# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.runner import Runner


class CustomEpochBasedRunner(Runner):
    """Custom runner that calls set_epoch on datasets each epoch."""

    def set_dataset(self, dataset):
        self._dataset = dataset

    def train(self, data_loader, **kwargs):
        if hasattr(self, '_dataset'):
            for dataset in self._dataset:
                if hasattr(dataset, 'set_epoch'):
                    dataset.set_epoch(self.epoch)
        super().train(data_loader, **kwargs)
