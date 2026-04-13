# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import re
import torch
from copy import deepcopy
from os import path as osp

from mmengine.config import Config
from mmengine.fileio import load as fileio_load
from mmengine.runner import load_checkpoint

from mmdet3d.core import (Box3DMode, CameraInstance3DBoxes,
                          DepthInstance3DBoxes, LiDARInstance3DBoxes,
                          show_multi_modality_result, show_result,
                          show_seg_result)
from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.models import build_model


def convert_SyncBN(config):
    """Convert config's naiveSyncBN to BN."""
    if isinstance(config, dict):
        for item in config:
            if item == 'norm_cfg':
                config[item]['type'] = config[item]['type']. \
                                    replace('naiveSyncBN', 'BN')
            else:
                convert_SyncBN(config[item])


def _get_test_pipeline_and_box_type(cfg):
    """Extract test pipeline and box_type_3d from either v1 or v2 config.

    v1 layout: cfg.data.test.pipeline / cfg.data.test.box_type_3d
    v2 layout: cfg.test_pipeline (top-level) or
               cfg.test_dataloader.dataset.pipeline /
               cfg.test_dataloader.dataset.box_type_3d
    """
    # --- pipeline ---
    if hasattr(cfg, 'data') and hasattr(cfg.data, 'test'):
        pipeline = deepcopy(cfg.data.test.pipeline)
        box_type_str = getattr(cfg.data.test, 'box_type_3d', 'LiDAR')
    elif hasattr(cfg, 'test_dataloader'):
        ds_cfg = cfg.test_dataloader.dataset
        # unwrap CBGSDataset or similar wrappers
        while hasattr(ds_cfg, 'dataset'):
            ds_cfg = ds_cfg.dataset
        pipeline = deepcopy(ds_cfg.pipeline)
        box_type_str = getattr(ds_cfg, 'box_type_3d', 'LiDAR')
    elif hasattr(cfg, 'test_pipeline'):
        pipeline = deepcopy(cfg.test_pipeline)
        box_type_str = 'LiDAR'
    else:
        raise AttributeError(
            'Cannot find test pipeline in config. Expected either '
            'cfg.data.test.pipeline (v1) or '
            'cfg.test_dataloader.dataset.pipeline (v2).')

    return pipeline, box_type_str


def init_model(config, checkpoint=None, device='cuda:0'):
    """Initialize a model from config file."""
    if isinstance(config, str):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    convert_SyncBN(config.model)
    config.model.train_cfg = None
    # Only pass top-level test_cfg if the model config doesn't already have one,
    # otherwise build_detector asserts "test_cfg specified in both".
    test_cfg = config.get('test_cfg') if not config.model.get('test_cfg') else None
    model = build_model(config.model, test_cfg=test_cfg)
    if checkpoint is not None:
        ckpt = load_checkpoint(model, checkpoint)
        if 'CLASSES' in ckpt.get('meta', {}):
            model.CLASSES = ckpt['meta']['CLASSES']
        else:
            model.CLASSES = config.class_names
        if 'PALETTE' in ckpt.get('meta', {}):
            model.PALETTE = ckpt['meta']['PALETTE']
    model.cfg = config
    model.to(device)
    model.eval()
    return model


def _prepare_data(data, device):
    """Move data to device, handling nested dicts, lists, and DataContainer."""
    from mmdet3d.compat import DataContainer
    if isinstance(data, DataContainer):
        return _prepare_data(data._data, device)
    elif isinstance(data, dict):
        return {k: _prepare_data(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(_prepare_data(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    return data


def inference_detector(model, pcd):
    """Inference point cloud with the detector."""
    cfg = model.cfg
    device = next(model.parameters()).device
    pipeline_cfg, box_type_str = _get_test_pipeline_and_box_type(cfg)
    test_pipeline = Compose(pipeline_cfg)
    box_type_3d, box_mode_3d = get_box_type(box_type_str)
    data = dict(
        pts_filename=pcd,
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        ann_info=dict(axis_align_matrix=np.eye(4)),
        sweeps=[],
        timestamp=[0],
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[])
    data = test_pipeline(data)
    data = _prepare_data(data, device)
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result, data


def inference_multi_modality_detector(model, pcd, image, ann_file):
    """Inference point cloud with the multi-modality detector."""
    cfg = model.cfg
    device = next(model.parameters()).device
    pipeline_cfg, box_type_str = _get_test_pipeline_and_box_type(cfg)
    test_pipeline = Compose(pipeline_cfg)
    box_type_3d, box_mode_3d = get_box_type(box_type_str)
    data_infos = fileio_load(ann_file)
    image_idx = int(re.findall(r'\d+', image)[-1])
    for x in data_infos:
        if int(x['image']['image_idx']) != image_idx:
            continue
        info = x
        break
    data = dict(
        pts_filename=pcd,
        img_prefix=osp.dirname(image),
        img_info=dict(filename=osp.basename(image)),
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[])
    data = test_pipeline(data)

    if box_mode_3d == Box3DMode.LIDAR:
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P2 = info['calib']['P2'].astype(np.float32)
        lidar2img = P2 @ rect @ Trv2c
        if 'img_metas' in data:
            data['img_metas'][0].data['lidar2img'] = lidar2img
    elif box_mode_3d == Box3DMode.DEPTH:
        rt_mat = info['calib']['Rt']
        rt_mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]
                           ]) @ rt_mat.transpose(1, 0)
        depth2img = info['calib']['K'] @ rt_mat
        if 'img_metas' in data:
            data['img_metas'][0].data['depth2img'] = depth2img

    data = _prepare_data(data, device)
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result, data


def inference_mono_3d_detector(model, image, ann_file):
    """Inference image with the monocular 3D detector."""
    cfg = model.cfg
    device = next(model.parameters()).device
    pipeline_cfg, box_type_str = _get_test_pipeline_and_box_type(cfg)
    test_pipeline = Compose(pipeline_cfg)
    box_type_3d, box_mode_3d = get_box_type(box_type_str)
    data_infos = fileio_load(ann_file)
    for x in data_infos['images']:
        if osp.basename(x['file_name']) != osp.basename(image):
            continue
        img_info = x
        break
    data = dict(
        img_prefix=osp.dirname(image),
        img_info=dict(filename=osp.basename(image)),
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[])

    if box_mode_3d == Box3DMode.CAM:
        data['img_info'].update(dict(cam_intrinsic=img_info['cam_intrinsic']))

    data = test_pipeline(data)
    data = _prepare_data(data, device)
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result, data


def inference_segmentor(model, pcd):
    """Inference point cloud with the segmentor."""
    cfg = model.cfg
    device = next(model.parameters()).device
    pipeline_cfg, _ = _get_test_pipeline_and_box_type(cfg)
    test_pipeline = Compose(pipeline_cfg)
    data = dict(
        pts_filename=pcd,
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[])
    data = test_pipeline(data)
    data = _prepare_data(data, device)
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result, data


def show_det_result_meshlab(data,
                            result,
                            out_dir,
                            score_thr=0.0,
                            show=False,
                            snapshot=False):
    """Show 3D detection result by meshlab."""
    points = data['points'][0][0].cpu().numpy()
    pts_filename = data['img_metas'][0][0]['pts_filename']
    file_name = osp.split(pts_filename)[-1].split('.')[0]

    if 'pts_bbox' in result[0].keys():
        pred_bboxes = result[0]['pts_bbox']['boxes_3d'].tensor.numpy()
        pred_scores = result[0]['pts_bbox']['scores_3d'].numpy()
    else:
        pred_bboxes = result[0]['boxes_3d'].tensor.numpy()
        pred_scores = result[0]['scores_3d'].numpy()

    if score_thr > 0:
        inds = pred_scores > score_thr
        pred_bboxes = pred_bboxes[inds]

    box_mode = data['img_metas'][0][0]['box_mode_3d']
    if box_mode != Box3DMode.DEPTH:
        points = points[..., [1, 0, 2]]
        points[..., 0] *= -1
        show_bboxes = Box3DMode.convert(pred_bboxes, box_mode, Box3DMode.DEPTH)
    else:
        show_bboxes = deepcopy(pred_bboxes)

    show_result(
        points,
        None,
        show_bboxes,
        out_dir,
        file_name,
        show=show,
        snapshot=snapshot)

    return file_name


def show_seg_result_meshlab(data,
                            result,
                            out_dir,
                            palette,
                            show=False,
                            snapshot=False):
    """Show 3D segmentation result by meshlab."""
    points = data['points'][0][0].cpu().numpy()
    pts_filename = data['img_metas'][0][0]['pts_filename']
    file_name = osp.split(pts_filename)[-1].split('.')[0]

    pred_seg = result[0]['semantic_mask'].numpy()

    if palette is None:
        max_idx = pred_seg.max()
        palette = np.random.randint(0, 256, size=(max_idx + 1, 3))
    palette = np.array(palette).astype(np.int32)

    show_seg_result(
        points,
        None,
        pred_seg,
        out_dir,
        file_name,
        palette=palette,
        show=show,
        snapshot=snapshot)

    return file_name


def show_proj_det_result_meshlab(data,
                                 result,
                                 out_dir,
                                 score_thr=0.0,
                                 show=False,
                                 snapshot=False):
    """Show result of projecting 3D bbox to 2D image by meshlab."""
    import mmcv
    assert 'img' in data.keys(), 'image data is not provided for visualization'

    img_filename = data['img_metas'][0][0]['filename']
    file_name = osp.split(img_filename)[-1].split('.')[0]

    img = mmcv.imread(img_filename)

    if 'pts_bbox' in result[0].keys():
        result[0] = result[0]['pts_bbox']
    elif 'img_bbox' in result[0].keys():
        result[0] = result[0]['img_bbox']
    pred_bboxes = result[0]['boxes_3d'].tensor.numpy()
    pred_scores = result[0]['scores_3d'].numpy()

    if score_thr > 0:
        inds = pred_scores > score_thr
        pred_bboxes = pred_bboxes[inds]

    box_mode = data['img_metas'][0][0]['box_mode_3d']
    if box_mode == Box3DMode.LIDAR:
        if 'lidar2img' not in data['img_metas'][0][0]:
            raise NotImplementedError(
                'LiDAR to image transformation matrix is not provided')

        show_bboxes = LiDARInstance3DBoxes(pred_bboxes, origin=(0.5, 0.5, 0))

        show_multi_modality_result(
            img,
            None,
            show_bboxes,
            data['img_metas'][0][0]['lidar2img'],
            out_dir,
            file_name,
            box_mode='lidar',
            show=show)
    elif box_mode == Box3DMode.DEPTH:
        show_bboxes = DepthInstance3DBoxes(pred_bboxes, origin=(0.5, 0.5, 0))

        show_multi_modality_result(
            img,
            None,
            show_bboxes,
            None,
            out_dir,
            file_name,
            box_mode='depth',
            img_metas=data['img_metas'][0][0],
            show=show)
    elif box_mode == Box3DMode.CAM:
        if 'cam2img' not in data['img_metas'][0][0]:
            raise NotImplementedError(
                'camera intrinsic matrix is not provided')

        show_bboxes = CameraInstance3DBoxes(
            pred_bboxes, box_dim=pred_bboxes.shape[-1], origin=(0.5, 1.0, 0.5))

        show_multi_modality_result(
            img,
            None,
            show_bboxes,
            data['img_metas'][0][0]['cam2img'],
            out_dir,
            file_name,
            box_mode='camera',
            show=show)
    else:
        raise NotImplementedError(
            f'visualization of {box_mode} bbox is not supported')

    return file_name


def show_result_meshlab(data,
                        result,
                        out_dir,
                        score_thr=0.0,
                        show=False,
                        snapshot=False,
                        task='det',
                        palette=None):
    """Show result by meshlab."""
    assert task in ['det', 'multi_modality-det', 'seg', 'mono-det'], \
        f'unsupported visualization task {task}'
    assert out_dir is not None, 'Expect out_dir, got none.'

    if task in ['det', 'multi_modality-det']:
        file_name = show_det_result_meshlab(data, result, out_dir, score_thr,
                                            show, snapshot)

    if task in ['seg']:
        file_name = show_seg_result_meshlab(data, result, out_dir, palette,
                                            show, snapshot)

    if task in ['multi_modality-det', 'mono-det']:
        file_name = show_proj_det_result_meshlab(data, result, out_dir,
                                                 score_thr, show, snapshot)

    return out_dir, file_name
