# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple

from mmengine.evaluator import BaseMetric
from mmengine.logging import print_log
from mmengine.registry import METRICS


@METRICS.register_module()
class IndoorMetric(BaseMetric):
    """Indoor 3D detection evaluation metric.

    Delegates to the dataset's ``evaluate()`` method (e.g. SUNRGBDDataset,
    ScanNetDataset, S3DISDataset).

    Args:
        metric (str or None): Metric name. Defaults to None.
        iou_thr (tuple[float]): IoU thresholds. Defaults to (0.25, 0.5).
        collect_device (str): Device for collecting results. Defaults to 'cpu'.
        prefix (str or None): Metric prefix. Defaults to None.
    """

    def __init__(self,
                 metric: Optional[str] = None,
                 iou_thr: Tuple[float, ...] = (0.25, 0.5),
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.metric = metric
        self.iou_thr = iou_thr

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

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        try:
            dataset = self.dataset
        except AttributeError:
            dataset = None

        if dataset is not None and hasattr(dataset, 'evaluate'):
            return dataset.evaluate(
                results, metric=self.metric, iou_thr=self.iou_thr)

        print_log(
            'IndoorMetric: Dataset.evaluate() not available. '
            'Cannot compute indoor detection metrics.',
            logger='current', level=40)
        raise RuntimeError(
            'IndoorMetric requires the dataset to provide an evaluate() '
            'method. Ensure the dataloader dataset is an indoor dataset '
            '(e.g. SUNRGBDDataset, ScanNetDataset, S3DISDataset).')
