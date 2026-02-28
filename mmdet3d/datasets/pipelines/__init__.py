# Copyright (c) OpenMMLab. All rights reserved.
# ----------------- 物理修复开始 -----------------
try:
    # 首先尝试旧版路径 (MMDet 2.x)
    from mmdet.datasets.pipelines import Compose
except (ImportError, ModuleNotFoundError):
    try:
        # 尝试 MMEngine 路径 (OpenMMLab 2.0 规范)
        from mmengine.dataset import Compose
    except (ImportError, ModuleNotFoundError):
        # 最后的保底路径 (MMDet 3.x 变换)
        from mmdet.datasets.transforms import Compose
# ----------------- 物理修复结束 -----------------
from .dbsampler import DataBaseSampler
from .formating import Collect3D, DefaultFormatBundle, DefaultFormatBundle3D
from .loading import (LoadAnnotations3D, LoadImageFromFileMono3D,
                      LoadMultiViewImageFromFiles, LoadPointsFromFile,
                      LoadPointsFromMultiSweeps, NormalizePointsColor,
                      PointSegClassMapping, LoadMultiViewImageFromFilesWaymo)
from .test_time_aug import MultiScaleFlipAug3D
from .transforms_3d import (BackgroundPointsFilter, GlobalAlignment,
                            GlobalRotScaleTrans, IndoorPatchPointSample,
                            IndoorPointSample, ObjectNameFilter, ObjectNoise,
                            ObjectRangeFilter, ObjectSample, PointSample,
                            PointShuffle, PointsRangeFilter,
                            RandomDropPointsColor, RandomFlip3D,
                            RandomJitterPoints, VoxelBasedPointSampler,
                            ObjectSampleV2, ResizeV2, PadV2, NormalizeV2,
                            PhotoMetricDistortionMultiViewImage, PadMultiViewImage,
                            NormalizeMultiviewImage, ScaleImageMultiViewImage)


__all__ = [
    'ObjectSample', 'RandomFlip3D', 'ObjectNoise', 'GlobalRotScaleTrans',
    'PointShuffle', 'ObjectRangeFilter', 'PointsRangeFilter', 'Collect3D',
    'Compose', 'LoadMultiViewImageFromFiles', 'LoadPointsFromFile',
    'DefaultFormatBundle', 'DefaultFormatBundle3D', 'DataBaseSampler',
    'NormalizePointsColor', 'LoadAnnotations3D', 'IndoorPointSample',
    'PointSample', 'PointSegClassMapping', 'MultiScaleFlipAug3D',
    'LoadPointsFromMultiSweeps', 'BackgroundPointsFilter',
    'VoxelBasedPointSampler', 'GlobalAlignment', 'IndoorPatchPointSample',
    'LoadImageFromFileMono3D', 'ObjectNameFilter', 'RandomDropPointsColor',
    'RandomJitterPoints', 'ObjectSampleV2', 'ResizeV2', 'PadV2', 'NormalizeV2', 
    'LoadMultiViewImageFromFilesWaymo', 'PadMultiViewImage', 'NormalizeMultiviewImage', 'PhotoMetricDistortionMultiViewImage', 'ScaleImageMultiViewImage',
]
