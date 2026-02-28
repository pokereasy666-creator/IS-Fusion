# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
# ----------------- 物理修复：猴子补丁补全 mmcv.is_list_of -----------------
if not hasattr(mmcv, 'is_list_of'):
    try:
        from mmengine.utils import is_list_of
        mmcv.is_list_of = is_list_of
    except ImportError:
        def is_list_of(seq, expected_type):
            return isinstance(seq, list) and all(isinstance(item, expected_type) for item in seq)
        mmcv.is_list_of = is_list_of
# ----------------- 物理修复结束 -----------------
import warnings
from copy import deepcopy

# ----------------- 物理修复开始 -----------------
try:
    # 尝试旧路径 (MMDet 2.x)
    from mmdet.registry import TRANSFORMS as PIPELINES
except ImportError:
    # 针对 MMDet 3.x 的新路径
    try:
        from mmdet.registry import PIPELINES
    except ImportError:
        # 保底方案：从 mmengine 导入通用注册表
        from mmengine.registry import Registry
        PIPELINES = Registry('pipeline')
# ----------------- 物理修复结束 -----------------
# ----------------- 物理修复开始 -----------------
try:
    from mmdet.datasets.pipelines import Compose
except (ImportError, ModuleNotFoundError):
    # 针对 MMEngine / MMCV 2.x 的新路径
    from mmengine.dataset import Compose
# ----------------- 物理修复结束 -----------------


@PIPELINES.register_module()
class MultiScaleFlipAug3D(object):
    """Test-time augmentation with multiple scales and flipping.

    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        img_scale (tuple | list[tuple]: Images scales for resizing.
        pts_scale_ratio (float | list[float]): Points scale ratios for
            resizing.
        flip (bool): Whether apply flip augmentation. Defaults to False.
        flip_direction (str | list[str]): Flip augmentation directions
            for images, options are "horizontal" and "vertical".
            If flip_direction is list, multiple flip augmentations will
            be applied. It has no effect when ``flip == False``.
            Defaults to "horizontal".
        pcd_horizontal_flip (bool): Whether apply horizontal flip augmentation
            to point cloud. Defaults to True. Note that it works only when
            'flip' is turned on.
        pcd_vertical_flip (bool): Whether apply vertical flip augmentation
            to point cloud. Defaults to True. Note that it works only when
            'flip' is turned on.
    """

    def __init__(self,
                 transforms,
                 img_scale,
                 pts_scale_ratio,
                 flip=False,
                 flip_direction='horizontal',
                 pcd_horizontal_flip=False,
                 pcd_vertical_flip=False):
                 # ----------------- 物理修复：终极注册表同步术 -----------------
        try:
            from mmengine.registry import TRANSFORMS
            # 1. 尝试同步 mmdet 的旧注册表
            try:
                from mmdet.datasets.builder import PIPELINES as DET_PIPELINES
                for name, obj in DET_PIPELINES.module_dict.items():
                    if not TRANSFORMS.get(name): 
                        TRANSFORMS.register_module(name=name, module=obj, force=True)
            except Exception: pass
            
            # 2. 尝试同步 mmdet3d 的旧注册表
            try:
                from mmdet3d.datasets.builder import PIPELINES as DET3D_PIPELINES
                for name, obj in DET3D_PIPELINES.module_dict.items():
                    if not TRANSFORMS.get(name): 
                        TRANSFORMS.register_module(name=name, module=obj, force=True)
            except Exception: pass
        except Exception as e:
            print(f"[Warning] Ultimate pipeline sync failed: {e}")
        # ----------------- 物理修复结束 -----------------
        self.transforms = Compose(transforms)
        self.img_scale = img_scale if isinstance(img_scale,
                                                 list) else [img_scale]
        self.pts_scale_ratio = pts_scale_ratio \
            if isinstance(pts_scale_ratio, list) else[float(pts_scale_ratio)]

        assert mmcv.is_list_of(self.img_scale, tuple)
        assert mmcv.is_list_of(self.pts_scale_ratio, float)

        self.flip = flip
        self.pcd_horizontal_flip = pcd_horizontal_flip
        self.pcd_vertical_flip = pcd_vertical_flip

        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]
        assert mmcv.is_list_of(self.flip_direction, str)
        if not self.flip and self.flip_direction != ['horizontal']:
            warnings.warn(
                'flip_direction has no effect when flip is set to False')
        if (self.flip and not any([(t['type'] == 'RandomFlip3D'
                                    or t['type'] == 'RandomFlip')
                                   for t in transforms])):
            warnings.warn(
                'flip has no effect when RandomFlip is not in transforms')

    def __call__(self, results):
        """Call function to augment common fields in results.

        Args:
            results (dict): Result dict contains the data to augment.

        Returns:
            dict: The result dict contains the data that is augmented with \
                different scales and flips.
        """
        aug_data = []

        # modified from `flip_aug = [False, True] if self.flip else [False]`
        # to reduce unnecessary scenes when using double flip augmentation
        # during test time
        flip_aug = [True] if self.flip else [False]
        pcd_horizontal_flip_aug = [False, True] \
            if self.flip and self.pcd_horizontal_flip else [False]
        pcd_vertical_flip_aug = [False, True] \
            if self.flip and self.pcd_vertical_flip else [False]
        for scale in self.img_scale:
            for pts_scale_ratio in self.pts_scale_ratio:
                for flip in flip_aug:
                    for pcd_horizontal_flip in pcd_horizontal_flip_aug:
                        for pcd_vertical_flip in pcd_vertical_flip_aug:
                            for direction in self.flip_direction:
                                # results.copy will cause bug
                                # since it is shallow copy
                                _results = deepcopy(results)
                                _results['scale'] = scale
                                _results['flip'] = flip
                                _results['pcd_scale_factor'] = \
                                    pts_scale_ratio
                                _results['flip_direction'] = direction
                                _results['pcd_horizontal_flip'] = \
                                    pcd_horizontal_flip
                                _results['pcd_vertical_flip'] = \
                                    pcd_vertical_flip
                                data = self.transforms(_results)
                                aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'img_scale={self.img_scale}, flip={self.flip}, '
        repr_str += f'pts_scale_ratio={self.pts_scale_ratio}, '
        repr_str += f'flip_direction={self.flip_direction})'
        return repr_str
