# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import torch
# ----------------- 物理修复：针对 DataContainer 路径迁移 -----------------
try:
    from mmcv.parallel import DataContainer as DC
except ImportError:
    try:
        # 适配 MMEngine 2.x 路径
        from mmcv.parallel import DataContainer as DC
    except ImportError:
        try:
            from mmcv.parallel import DataContainer as DC
        except ImportError:
            # 最后的保底：如果实在找不到，定义一个简单的类避免代码完全崩溃
            class DC:
                def __init__(self, data, **kwargs): self.data = data
# ----------------- 物理修复结束 -----------------
# ----------------- 物理修复：针对 mmcv.runner 装饰器迁移 -----------------
try:
    from mmcv.runner import auto_fp16
except ImportError:
    # 适配 MMEngine / MMCV 2.x 
    # MMEngine 中通常不再强制要求 auto_fp16 装饰器，这里提供一个空装饰器保底
    def auto_fp16(apply_to=None, out_fp16=False):
        def decorator(func):
            return func
        return decorator

# 如果后面有 BaseModule 报错，可以提前加上这个探测
if 'BaseModule' not in globals():
    try:
        from mmcv.runner import BaseModule
    except ImportError:
        import torch.nn as nn
        BaseModule = nn.Module
# ----------------- 物理修复结束 -----------------
from os import path as osp

from mmdet3d.core import show_seg_result
from mmseg.models.segmentors import BaseSegmentor


class Base3DSegmentor(BaseSegmentor):
    """Base class for 3D segmentors.

    The main difference with `BaseSegmentor` is that we modify the keys in
    data_dict and use a 3D seg specific visualization function.
    """

    @property
    def with_regularization_loss(self):
        """bool: whether the segmentor has regularization loss for weight"""
        return hasattr(self, 'loss_regularization') and \
            self.loss_regularization is not None

    def forward_test(self, points, img_metas, **kwargs):
        """Calls either simple_test or aug_test depending on the length of
        outer list of points. If len(points) == 1, call simple_test. Otherwise
        call aug_test to aggregate the test results by e.g. voting.

        Args:
            points (list[list[torch.Tensor]]): the outer list indicates
                test-time augmentations and inner torch.Tensor should have a
                shape BXNxC, which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(points, 'points'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(points)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(points)}) != '
                             f'num of image meta ({len(img_metas)})')

        if num_augs == 1:
            return self.simple_test(points[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(points, img_metas, **kwargs)

    @auto_fp16(apply_to=('points'))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, point and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, point and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def show_results(self,
                     data,
                     result,
                     palette=None,
                     out_dir=None,
                     ignore_index=None):
        """Results visualization.

        Args:
            data (list[dict]): Input points and the information of the sample.
            result (list[dict]): Prediction results.
            palette (list[list[int]]] | np.ndarray | None): The palette of
                segmentation map. If None is given, random palette will be
                generated. Default: None
            out_dir (str): Output directory of visualization result.
            ignore_index (int, optional): The label index to be ignored, e.g.
                unannotated points. If None is given, set to len(self.CLASSES).
                Defaults to None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        if palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(
                    0, 255, size=(len(self.CLASSES), 3))
            else:
                palette = self.PALETTE
        palette = np.array(palette)
        for batch_id in range(len(result)):
            if isinstance(data['points'][0], DC):
                points = data['points'][0]._data[0][batch_id].numpy()
            elif mmcv.is_list_of(data['points'][0], torch.Tensor):
                points = data['points'][0][batch_id]
            else:
                ValueError(f"Unsupported data type {type(data['points'][0])} "
                           f'for visualization!')
            if isinstance(data['img_metas'][0], DC):
                pts_filename = data['img_metas'][0]._data[0][batch_id][
                    'pts_filename']
            elif mmcv.is_list_of(data['img_metas'][0], dict):
                pts_filename = data['img_metas'][0][batch_id]['pts_filename']
            else:
                ValueError(
                    f"Unsupported data type {type(data['img_metas'][0])} "
                    f'for visualization!')
            file_name = osp.split(pts_filename)[-1].split('.')[0]

            pred_sem_mask = result[batch_id]['semantic_mask'].cpu().numpy()

            show_seg_result(
                points,
                None,
                pred_sem_mask,
                out_dir,
                file_name,
                palette,
                ignore_index,
                show=True)
