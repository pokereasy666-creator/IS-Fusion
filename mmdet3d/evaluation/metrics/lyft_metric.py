# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence

from mmengine.evaluator import BaseMetric
from mmengine.logging import print_log
from mmengine.registry import METRICS


@METRICS.register_module()
class LyftMetric(BaseMetric):
    """Lyft evaluation metric.

    Delegates to ``LyftDataset.evaluate()`` for the full Lyft evaluation
    protocol.

    Args:
        metric (str): Metric name. Defaults to 'bbox'.
        collect_device (str): Device for collecting results. Defaults to 'cpu'.
        prefix (str or None): Metric prefix. Defaults to None.
    """

    def __init__(self,
                 metric: str = 'bbox',
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
            'LyftMetric: Dataset.evaluate() not available. '
            'Cannot compute Lyft metrics.',
            logger='current', level=40)
        raise RuntimeError(
            'LyftMetric requires the dataset to provide an evaluate() '
            'method. Ensure the dataloader dataset is a LyftDataset and '
            'add DatasetInjectionHook to custom_hooks in your config.')
