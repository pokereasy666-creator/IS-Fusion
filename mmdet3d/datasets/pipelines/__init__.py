# Copyright (c) OpenMMLab. All rights reserved.
from mmdet3d.compat import build_from_cfg

try:
    from mmdet.datasets.pipelines import Compose
except (ImportError, ModuleNotFoundError):
    try:
        from mmengine.dataset import Compose
    except ImportError:
        from mmdet.datasets.transforms import Compose
from .dbsampler import DataBaseSampler
from .formating import Collect3D, Collect3DV2, DefaultFormatBundle, DefaultFormatBundle3D
from .loading import (LoadAnnotations3D, LoadImageFromFileMono3D,
                      LoadMultiViewImageFromFiles,
                      LoadMultiViewImageFromFilesV2, LoadPointsFromFile,
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
    'Collect3DV2', 'Compose', 'LoadMultiViewImageFromFiles',
    'LoadMultiViewImageFromFilesV2', 'LoadPointsFromFile',
    'DefaultFormatBundle', 'DefaultFormatBundle3D', 'DataBaseSampler',
    'NormalizePointsColor', 'LoadAnnotations3D', 'IndoorPointSample',
    'PointSample', 'PointSegClassMapping', 'MultiScaleFlipAug3D',
    'LoadPointsFromMultiSweeps', 'BackgroundPointsFilter',
    'VoxelBasedPointSampler', 'GlobalAlignment', 'IndoorPatchPointSample',
    'LoadImageFromFileMono3D', 'ObjectNameFilter', 'RandomDropPointsColor',
    'RandomJitterPoints', 'ObjectSampleV2', 'ResizeV2', 'PadV2', 'NormalizeV2',
    'LoadMultiViewImageFromFilesWaymo', 'PadMultiViewImage',
    'NormalizeMultiviewImage', 'PhotoMetricDistortionMultiViewImage',
    'ScaleImageMultiViewImage',
]
