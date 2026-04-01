# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.registry import HOOKS


@HOOKS.register_module()
class DatasetInjectionHook(Hook):
    """Hook that injects the dataset instance into evaluator metrics.

    mmengine's ``Evaluator`` does not automatically bind the dataset object
    to metric instances.  This hook runs before each validation/test epoch
    and calls ``metric.set_dataset(dataset)`` (if available) on every metric
    registered in the evaluator, so that metrics which delegate to
    ``dataset.evaluate()`` can function correctly.

    Add this hook to ``custom_hooks`` in your config::

        custom_hooks = [dict(type='DatasetInjectionHook')]
    """

    priority = 'ABOVE_NORMAL'

    def _inject(self, runner, dataloader_attr, evaluator_attr):
        dataloader = getattr(runner, dataloader_attr, None)
        evaluator = getattr(runner, evaluator_attr, None)
        if dataloader is None or evaluator is None:
            return
        dataset = getattr(dataloader, 'dataset', None)
        if dataset is None:
            return
        # Walk through dataset wrappers (e.g. CBGSDataset)
        while hasattr(dataset, 'dataset') and not hasattr(dataset, 'evaluate'):
            dataset = dataset.dataset
        metrics = getattr(evaluator, 'metrics', [])
        for metric in metrics:
            if hasattr(metric, 'set_dataset'):
                metric.set_dataset(dataset)

    def before_val_epoch(self, runner):
        self._inject(runner, 'val_dataloader', 'val_evaluator')

    def before_test_epoch(self, runner):
        self._inject(runner, 'test_dataloader', 'test_evaluator')
