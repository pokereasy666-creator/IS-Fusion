# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from typing import Dict, List, Optional, Sequence

from mmengine.evaluator import BaseMetric
from mmengine.logging import print_log
from mmengine.registry import METRICS


@METRICS.register_module()
class NuScenesMetric(BaseMetric):
    """nuScenes evaluation metric.

    Wraps the nuScenes detection evaluation protocol into an MMEngine-style
    metric so it can be used with ``Runner.from_cfg()``.

    Args:
        data_root (str): Root directory of the nuScenes dataset.
            Defaults to ``'data/nuscenes/'``.
        metric (str): Metric name. Defaults to ``'bbox'``.
        jsonfile_prefix (str or None): Prefix of the json result file.
            If None a temp dir will be used. Defaults to None.
        collect_device (str): Device for collecting results from all
            ranks. Defaults to ``'cpu'``.
        prefix (str or None): Metric prefix. Defaults to None.
    """

    def __init__(self,
                 data_root: str = 'data/nuscenes/',
                 metric: str = 'bbox',
                 jsonfile_prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.data_root = data_root
        self.metric = metric
        self.jsonfile_prefix = jsonfile_prefix

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Collect prediction results from *data_samples*.

        Each element in *data_samples* is expected to carry either
        ``pred_instances_3d`` (standard mmdet3d v2) **or** the legacy
        dict format with keys like ``pts_bbox`` / ``boxes_3d``.
        """
        for data_sample in data_samples:
            result = dict()
            if hasattr(data_sample, 'pred_instances_3d'):
                pred = data_sample.pred_instances_3d
                result['pts_bbox'] = dict(
                    boxes_3d=pred.bboxes_3d,
                    scores_3d=pred.scores_3d,
                    labels_3d=pred.labels_3d)
            elif isinstance(data_sample, dict):
                result = data_sample
            else:
                result = data_sample.to_dict()
            self.results.append(result)

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute nuScenes detection metrics.

        Delegates to the ``evaluate()`` method on the dataset object attached
        to the runner's test dataloader, which already implements the full
        nuScenes evaluation protocol via the nuscenes-devkit.
        """
        dataset = self._get_dataset()
        if dataset is not None and hasattr(dataset, 'evaluate'):
            return dataset.evaluate(
                results,
                metric=self.metric,
                jsonfile_prefix=self.jsonfile_prefix)

        # Fallback: run the evaluation directly via nuscenes-devkit
        print_log(
            'Dataset.evaluate() not available, running nuscenes-devkit '
            'evaluation directly.',
            logger='current')
        return self._evaluate_via_devkit(results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_dataset(self):
        """Try to obtain the dataset from the runner.

        mmengine's Evaluator sets ``self.dataset`` on the metric when the
        evaluator is constructed with a dataloader. If it was never set,
        we return None instead of raising.
        """
        return getattr(self, 'dataset', None)

    def _evaluate_via_devkit(self, results: List[dict]) -> Dict[str, float]:
        """Minimal direct nuscenes-devkit evaluation."""
        try:
            from nuscenes import NuScenes
            from nuscenes.eval.detection.config import config_factory
            from nuscenes.eval.detection.evaluate import NuScenesEval
        except ImportError:
            print_log(
                'nuscenes-devkit is not installed. '
                'Cannot compute nuScenes metrics.',
                logger='current',
                level=40)
            raise RuntimeError(
                'NuScenesMetric: neither Dataset.evaluate() nor '
                'nuscenes-devkit is available. Install nuscenes-devkit or '
                'ensure the dataloader dataset is a NuScenesDataset.')

        tmp_dir = None
        if self.jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            jsonfile_prefix = self.jsonfile_prefix

        try:
            from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset
            # Use NuScenesDataset's format_results + nuScenes devkit
            result_files, _ = NuScenesDataset.format_results_static(
                results, jsonfile_prefix)
            if isinstance(result_files, dict):
                result_path = result_files.get('pts_bbox', result_files.get(
                    next(iter(result_files))))
            else:
                result_path = result_files

            nusc = NuScenes(
                version='v1.0-trainval', dataroot=self.data_root, verbose=False)
            eval_config = config_factory('detection_cvpr_2019')
            nusc_eval = NuScenesEval(
                nusc, config=eval_config, result_path=result_path,
                eval_set='val', output_dir=osp.dirname(jsonfile_prefix),
                verbose=False)
            metrics_summary = nusc_eval.main(render_curves=False)
            metrics = dict()
            for k, v in metrics_summary['label_aps'].items():
                for thresh, ap in v.items():
                    metrics[f'{k}_AP_{thresh}'] = ap
            metrics['NDS'] = metrics_summary.get('nd_score', 0.0)
            metrics['mAP'] = metrics_summary.get('mean_ap', 0.0)
            return metrics
        except Exception as e:
            print_log(
                f'Direct nuscenes-devkit evaluation failed: {e}',
                logger='current', level=40)
            raise
        finally:
            if tmp_dir is not None:
                tmp_dir.cleanup()
