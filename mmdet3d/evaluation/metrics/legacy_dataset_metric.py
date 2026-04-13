# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence

from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS


def _iter_evaluator_metrics(evaluator):
    metrics = getattr(evaluator, 'metrics', None)
    if metrics is None:
        metrics = getattr(evaluator, '_metrics', None)
    if metrics is None:
        return (evaluator,)
    if isinstance(metrics, dict):
        return tuple(metrics.values())
    if isinstance(metrics, (list, tuple)):
        return tuple(metrics)
    return (metrics,)


def attach_dataset_to_metrics(evaluator, dataset) -> None:
    if evaluator is None or dataset is None:
        return

    for metric in _iter_evaluator_metrics(evaluator):
        if getattr(metric, 'requires_dataset', False):
            metric.dataset = dataset


def attach_runner_datasets_to_metrics(runner, *splits) -> None:
    for split in splits:
        evaluator = getattr(runner, f'{split}_evaluator')
        dataloader = getattr(runner, f'{split}_dataloader')
        attach_dataset_to_metrics(evaluator, dataloader.dataset)


class LegacyDatasetMetric(BaseMetric):
    """MMEngine metric wrapper for legacy dataset ``evaluate`` methods."""

    default_metric = None
    requires_dataset = True

    def __init__(self,
                 metric=None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 **eval_kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.metric = self.default_metric if metric is None else metric
        self.eval_kwargs = eval_kwargs

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            self.results.append(self._to_legacy_result(data_sample))

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        dataset = self._dataset
        if dataset is None or not hasattr(dataset, 'evaluate'):
            raise RuntimeError(
                f'{self.__class__.__name__} requires the runner dataset to be '
                'attached to the metric and to implement evaluate().')

        kwargs = self.eval_kwargs.copy()
        if self.metric is not None:
            kwargs.setdefault('metric', self.metric)
        return dataset.evaluate(results, **kwargs)

    @property
    def _dataset(self):
        try:
            return self.dataset
        except AttributeError:
            return None

    @staticmethod
    def _to_legacy_result(data_sample):
        if isinstance(data_sample, dict):
            return data_sample
        if hasattr(data_sample, 'pred_instances_3d'):
            pred = data_sample.pred_instances_3d
            return dict(
                pts_bbox=dict(
                    boxes_3d=pred.bboxes_3d,
                    scores_3d=pred.scores_3d,
                    labels_3d=pred.labels_3d))
        if hasattr(data_sample, 'pred_pts_seg'):
            pred = data_sample.pred_pts_seg
            if hasattr(pred, 'pts_semantic_mask'):
                return dict(semantic_mask=pred.pts_semantic_mask)
        if hasattr(data_sample, 'to_dict'):
            return data_sample.to_dict()
        return data_sample


@METRICS.register_module()
class KittiMetric(LegacyDatasetMetric):
    pass


@METRICS.register_module()
class LyftMetric(LegacyDatasetMetric):
    default_metric = 'bbox'


@METRICS.register_module()
class WaymoMetric(LegacyDatasetMetric):
    default_metric = 'waymo'


@METRICS.register_module()
class IndoorMetric(LegacyDatasetMetric):
    @staticmethod
    def _to_legacy_result(data_sample):
        if isinstance(data_sample, dict):
            if 'pts_bbox' in data_sample:
                return data_sample['pts_bbox']
            return data_sample
        if hasattr(data_sample, 'pred_instances_3d'):
            pred = data_sample.pred_instances_3d
            return dict(
                boxes_3d=pred.bboxes_3d,
                scores_3d=pred.scores_3d,
                labels_3d=pred.labels_3d)
        return LegacyDatasetMetric._to_legacy_result(data_sample)


@METRICS.register_module()
class SegMetric(LegacyDatasetMetric):
    pass
