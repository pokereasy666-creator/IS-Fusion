# Copyright (c) OpenMMLab. All rights reserved.
# ----------------- 物理修复：手写智能拆包 Dataloader -----------------
import torch
from torch.utils.data import DataLoader

def custom_collate(batch):
    """专门用来生剥旧版 DataContainer 的智能打包器"""
    if not isinstance(batch, list):
        return batch
    
    elem = batch[0]
    # 1. 如果是字典，递归解包里面的每一个值
    if isinstance(elem, dict):
        return {key: custom_collate([d[key] for d in batch]) for key in elem}
    
    # 2. 【核心】如果碰到了旧版余孽 DataContainer，直接拆壳！
    elif type(elem).__name__ == 'DataContainer':
        # 如果它要求被堆叠 (stack)
        if getattr(elem, 'stack', False):
            data_list = [d.data for d in batch]
            if isinstance(data_list[0], torch.Tensor):
                # 检查尺寸是否一致，一致就 stack，不一致就保留 List
                if all(d.shape == data_list[0].shape for d in data_list):
                    return torch.stack(data_list, 0)
            return data_list
        # 如果不要求堆叠，直接剥离并组成 List 返回
        else:
            return [d.data for d in batch]
            
    # 3. 如果已经是纯 Tensor，按规矩打包
    elif isinstance(elem, torch.Tensor):
        if all(d.shape == elem.shape for d in batch):
            return torch.stack(batch, 0)
        return batch
        
    # 4. 其他基础类型直接放行
    elif isinstance(elem, (int, float)):
        return torch.tensor(batch)
    else:
        return batch

def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=False,
                     shuffle=False,
                     seed=None,
                     **kwargs):
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if dist else None
    data_loader = DataLoader(
        dataset,
        batch_size=samples_per_gpu,
        sampler=sampler,
        num_workers=workers_per_gpu,
        collate_fn=custom_collate,  # <--- 使用我们自己的拆包器
        pin_memory=kwargs.pop('pin_memory', False),
        shuffle=(shuffle if sampler is None else False),
        **kwargs)
    return data_loader
# ----------------- 物理修复结束 -----------------
from .builder import DATASETS, build_dataset
from .custom_3d import Custom3DDataset
from .custom_3d_seg import Custom3DSegDataset
from .kitti_dataset import KittiDataset
from .kitti_mono_dataset import KittiMonoDataset
from .lyft_dataset import LyftDataset
from .nuscenes_dataset import NuScenesDataset
from .nuscenes_mono_dataset import NuScenesMonoDataset
# yapf: disable
from .pipelines import (BackgroundPointsFilter, GlobalAlignment,
                        GlobalRotScaleTrans, IndoorPatchPointSample,
                        IndoorPointSample, LoadAnnotations3D,
                        LoadPointsFromFile, LoadPointsFromMultiSweeps,
                        NormalizePointsColor, ObjectNameFilter, ObjectNoise,
                        ObjectRangeFilter, ObjectSample, PointSample,
                        PointShuffle, PointsRangeFilter, RandomDropPointsColor,
                        RandomFlip3D, RandomJitterPoints,
                        VoxelBasedPointSampler)
# yapf: enable
from .s3dis_dataset import S3DISDataset, S3DISSegDataset
from .scannet_dataset import ScanNetDataset, ScanNetSegDataset
from .semantickitti_dataset import SemanticKITTIDataset
from .sunrgbd_dataset import SUNRGBDDataset
from .utils import get_loading_pipeline
from .waymo_dataset import WaymoDataset

__all__ = [
    'KittiDataset', 'KittiMonoDataset', 'build_dataloader', 'DATASETS',
    'build_dataset', 'NuScenesDataset', 'NuScenesMonoDataset', 'LyftDataset',
    'ObjectSample', 'RandomFlip3D', 'ObjectNoise', 'GlobalRotScaleTrans',
    'PointShuffle', 'ObjectRangeFilter', 'PointsRangeFilter',
    'LoadPointsFromFile', 'S3DISSegDataset', 'S3DISDataset',
    'NormalizePointsColor', 'IndoorPatchPointSample', 'IndoorPointSample',
    'PointSample', 'LoadAnnotations3D', 'GlobalAlignment', 'SUNRGBDDataset',
    'ScanNetDataset', 'ScanNetSegDataset', 'SemanticKITTIDataset',
    'Custom3DDataset', 'Custom3DSegDataset', 'LoadPointsFromMultiSweeps',
    'WaymoDataset', 'BackgroundPointsFilter', 'VoxelBasedPointSampler',
    'get_loading_pipeline', 'RandomDropPointsColor', 'RandomJitterPoints',
    'ObjectNameFilter'
]
