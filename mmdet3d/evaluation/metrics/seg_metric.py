# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence

from mmengine.evaluator import BaseMetric
from mmengine.logging import print_log
from mmengine.registry import METRICS


@METRICS.register_module()
class SegMetric(BaseMetric):
    """3D segmentation evaluation metric.

    Delegates to the dataset's ``evaluate()`` method (e.g.
    ScanNetSegDataset, S3DISSegDataset, SemanticKITTIDataset).

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
            if hasattr(data_sample, 'pred_pts_seg'):
                pred = data_sample.pred_pts_seg
                result = dict(semantic_mask=pred.pts_semantic_mask)
            elif isinstance(data_sample, dict):
                result = data_sample
            else:
                result = data_sample.to_dict()
            self.results.append(result)

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        try:
            dataset = self.dataset
        except AttributeError:
            dataset = None

        if dataset is not None and hasattr(dataset, 'evaluate'):
            return dataset.evaluate(results, metric=self.metric)

        print_log(
            'SegMetric: Dataset.evaluate() not available. '
            'Cannot compute segmentation metrics.',
            logger='current', level=40)
        raise RuntimeError(
            'SegMetric requires the dataset to provide an evaluate() '
            'method. Ensure the dataloader dataset is a segmentation dataset.')
