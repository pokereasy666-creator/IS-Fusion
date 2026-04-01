# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence

from mmengine.evaluator import BaseMetric
from mmengine.logging import print_log
from mmengine.registry import METRICS


@METRICS.register_module()
class KittiMetric(BaseMetric):
    """KITTI evaluation metric.

    Delegates to ``KittiDataset.evaluate()`` for the full KITTI evaluation
    protocol.

    Args:
        metric (str or None): Metric name. Defaults to None.
        collect_device (str): Device for collecting results. Defaults to 'cpu'.
        prefix (str or None): Metric prefix. Defaults to None.
    """

    def __init__(self,
                 metric: Optional[str] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.metric = metric

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            if hasattr(data_sample, 'pred_instances_3d'):
                pred = data_sample.pred_instances_3d
                result = dict(
                    boxes_3d=pred.bboxes_3d,
                    scores_3d=pred.scores_3d,
                    labels_3d=pred.labels_3d)
            elif isinstance(data_sample, dict):
                result = data_sample
            else:
                result = data_sample.to_dict()
            self.results.append(result)

    def set_dataset(self, dataset):
        """Explicitly inject the dataset instance for evaluation."""
        self.dataset = dataset

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        dataset = getattr(self, 'dataset', None)

        if dataset is not None and hasattr(dataset, 'evaluate'):
            return dataset.evaluate(results, metric=self.metric)

        print_log(
            'KittiMetric: Dataset.evaluate() not available. '
            'Cannot compute KITTI metrics.',
            logger='current', level=40)
        raise RuntimeError(
            'KittiMetric requires the dataset to provide an evaluate() '
            'method. Ensure the dataloader dataset is a KittiDataset and '
            'add DatasetInjectionHook to custom_hooks in your config.')
