# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import os
import os.path as osp
from PIL import Image
from typing import Any, Dict, Tuple

# [兼容性修复] 使用与 transforms_3d.py 相同的共享注册表
try:
    from mmdet.datasets.builder import PIPELINES
except ImportError:
    from mmcv.utils import Registry
    PIPELINES = Registry('pipeline')

from mmdet3d.core.points import BasePoints, get_points_type

# [兼容性修复] 尝试导入基类
try:
    # MMDet 2.x
    from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
except ImportError:
    try:
        # MMDet 3.x
        from mmdet.datasets.transforms import LoadAnnotations, LoadImageFromFile
    except ImportError:
        try:
            from mmcv.transforms import LoadImageFromFile
            from mmdet.datasets.transforms import LoadAnnotations
        except ImportError:
            class LoadImageFromFile: pass
            class LoadAnnotations: pass

@PIPELINES.register_module()
class LoadMultiViewImageFromFilesV2:  # v2: bevfusion
    def __init__(self, to_float32=False, color_type="unchanged"):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        if "img_filename" not in results:
            return results
        filename = results["img_filename"]
        images = []
        for name in filename:
            images.append(Image.open(name))
        results["filename"] = filename
        results["img"] = images
        results["img_shape"] = images[0].size
        results["ori_shape"] = images[0].size
        results["pad_shape"] = images[0].size
        results["scale_factor"] = 1.0
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(to_float32={self.to_float32}, color_type='{self.color_type}')"

@PIPELINES.register_module()
class LoadImageFromFileV2:
    def __init__(self, to_float32=False, color_type='color', file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        # [自动纠错] disk -> local
        if self.file_client_args.get('backend') == 'disk':
            self.file_client_args['backend'] = 'local'

    def __call__(self, results):
        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'], results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img = Image.open(filename)
        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = [img]
        results['img_shape'] = img.size
        results['ori_shape'] = img.size
        results['img_fields'] = ['img']
        results["pad_shape"] = img.size
        results["scale_factor"] = 1.0
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(to_float32={self.to_float32}, color_type='{self.color_type}')"

@PIPELINES.register_module()
class MyResize(object):
    def __init__(self, img_scale=None, multiscale_mode='range', ratio_range=None, keep_ratio=True, bbox_clip_border=True, backend='cv2', override=False):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
        if ratio_range is not None:
            assert len(self.img_scale) == 1
        else:
            assert multiscale_mode in ['value', 'range']
        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.override = override
        self.bbox_clip_border = bbox_clip_border

    @staticmethod
    def random_select(img_scales):
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(min(img_scale_long), max(img_scale_long) + 1)
        short_edge = np.random.randint(min(img_scale_short), max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError
        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        imgs = results['img']
        results['img'] = [imgs[i] for i in range(len(imgs))]
        for key in results.get('img_fields', ['img']):
            for idx in range(len(results['img'])):
                if self.keep_ratio:
                    img, scale_factor = mmcv.imrescale(results[key][idx], results['scale'], return_scale=True, backend=self.backend)
                    new_h, new_w = img.shape[:2]
                    h, w = results[key][idx].shape[:2]
                    w_scale = new_w / w
                    h_scale = new_h / h
                else:
                    img, w_scale, h_scale = mmcv.imresize(results[key][idx], results['scale'], return_scale=True, backend=self.backend)
                results[key][idx] = img
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale, 1.0], dtype=np.float32)
            results['img_shape'] = img.shape
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        for key in results.get('bbox_fields', []):
            bboxes = results[key] * results['scale_factor']
            if self.bbox_clip_border:
                img_shape = results['img_shape']
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            results[key] = bboxes

    def _resize_masks(self, results):
        for key in results.get('mask_fields', []):
            if results[key] is None: continue
            if self.keep_ratio:
                results[key] = results[key].rescale(results['scale'])
            else:
                results[key] = results[key].resize(results['img_shape'][:2])

    def _resize_seg(self, results):
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(results[key], results['scale'], interpolation='nearest', backend=self.backend)
            else:
                gt_seg = mmcv.imresize(results[key], results['scale'], interpolation='nearest', backend=self.backend)
            results['gt_semantic_seg'] = gt_seg

    def __call__(self, results):
        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['img'][0].shape[:2]
                scale_factor = results['scale_factor']
                assert isinstance(scale_factor, float)
                results['scale'] = tuple([int(x * scale_factor) for x in img_shape][::-1])
            else:
                self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, 'scale and scale_factor cannot be both set.'
            else:
                results.pop('scale')
                if 'scale_factor' in results: results.pop('scale_factor')
                self._random_scale(results)
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(img_scale={self.img_scale}, multiscale_mode={self.multiscale_mode})"

@PIPELINES.register_module()
class MyNormalize(object):
    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            for idx in range(len(results['img'])):
                results[key][idx] = mmcv.imnormalize(results[key][idx], self.mean, self.std, self.to_rgb)
        results['img_norm_cfg'] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results
    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"

@PIPELINES.register_module()
class MyPad(object):
    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None
    def _pad_img(self, results):
        for key in results.get('img_fields', ['img']):
            if self.size is not None:
                padded_img = mmcv.impad(results[key], shape=self.size, pad_val=self.pad_val)
            elif self.size_divisor is not None:
                for idx in range(len(results[key])):
                    padded_img = mmcv.impad_to_multiple(results[key][idx], self.size_divisor, pad_val=self.pad_val)
                    results[key][idx] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor
    def _pad_masks(self, results):
        pad_shape = results['pad_shape'][:2]
        for key in results.get('mask_fields', []):
            results[key] = results[key].pad(pad_shape, pad_val=self.pad_val)
    def _pad_seg(self, results):
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(results[key], shape=results['pad_shape'][:2])
    def __call__(self, results):
        self._pad_img(results)
        self._pad_masks(results)
        self._pad_seg(results)
        return results
    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size}, size_divisor={self.size_divisor}, pad_val={self.pad_val})"

@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    def __init__(self, to_float32=False, color_type='unchanged', **kwargs):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        filename = results['img_filename']
        img = np.stack([mmcv.imread(name, flag=self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_fields'] = ['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(mean=np.zeros(num_channels, dtype=np.float32), std=np.ones(num_channels, dtype=np.float32), to_rgb=False)
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(to_float32={self.to_float32}, color_type='{self.color_type}')"

@PIPELINES.register_module()
class LoadMultiViewImageFromFilesWaymo(object):
    def __init__(self, to_float32=False, color_type='unchanged', num_img=5):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.num_img = num_img
    def __call__(self, results):
        filename = results['img_info']['filename']
        img_list = []
        for name in filename:
            img = mmcv.imread(name, self.color_type)
            short_edge = img.shape[0]
            if short_edge != 1280:
                new_img = np.zeros((1280, 1920, 3))
                new_img[:short_edge] = img
                img = new_img
            img_list.append(img)
        img = np.stack(img_list, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_fields'] = ['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(mean=np.zeros(num_channels, dtype=np.float32), std=np.ones(num_channels, dtype=np.float32), to_rgb=False)
        return results
    def __repr__(self):
        return f"{self.__class__.__name__}(to_float32={self.to_float32})"

@PIPELINES.register_module()
class LoadImageFromFileMono3D(LoadImageFromFile):
    def __call__(self, results):
        super().__call__(results)
        results['cam2img'] = results['img_info']['cam_intrinsic']
        return results

@PIPELINES.register_module()
class LoadPointsFromMultiSweeps(object):
    def __init__(self, sweeps_num=10, load_dim=5, use_dim=[0, 1, 2, 4], file_client_args=dict(backend='disk'), pad_empty_sweeps=False, remove_close=False, test_mode=False, painting=False):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        # [自动纠错] disk -> local
        if self.file_client_args.get('backend') == 'disk':
            self.file_client_args['backend'] = 'local'
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.painting = painting
        if self.painting:
            self.predict_fun = paint2seg_func()

    def _load_points(self, pts_filename):
        try:
            pts_bytes = mmcv.load(pts_filename, backend_args=self.file_client_args)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except Exception: 
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _load_painting(self, points, painting_path):
        try:
            painting_bytes = mmcv.load(painting_path, backend_args=self.file_client_args)
            predict_idx = np.frombuffer(painting_bytes, dtype='uint8')
        except Exception:
             predict_idx = np.fromfile(painting_path, dtype='uint8')
        predict_idx_2d = self.predict_fun(predict_idx)
        sample_predict_label = np.eye(11, dtype='uint8')[predict_idx_2d]
        points = np.concatenate([points, sample_predict_label], axis=1)
        return points

    def _remove_close(self, points, radius=1.0):
        if isinstance(points, np.ndarray): points_numpy = points
        elif isinstance(points, BasePoints): points_numpy = points.tensor.numpy()
        else: raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        points = results['points']
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close: sweep_points_list.append(self._remove_close(points))
                else: sweep_points_list.append(points)
        else:
            if len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close: points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep['sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, 4] = ts - sweep_ts
                if self.painting:
                    painting_path = sweep['data_path'].split('/')
                    painting_path[-2] = 'LIDAR_TOP_MASK'
                    painting_path = '/'.join(i for i in painting_path)
                    if osp.isfile(painting_path): points_sweep = self._load_painting(points_sweep, painting_path)
                    else: continue
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)
        points = points.cat(sweep_points_list)
        if not self.painting: points = points[:, self.use_dim]
        results['points'] = points
        return results
    def __repr__(self):
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'

@PIPELINES.register_module()
class LoadForeground2D(object):
    def __init__(self, dataset='NuScenesDataset', **kwargs):
        self.dataset = dataset
    def _organize(self, fg_info):
        if self.dataset == 'NuScenesDataset':
            cam_num = len(fg_info['virtual_pixel_indices'])
            fg_pixels, fg_points, fg_real_pixels, fg_real_points = [], [], [], []
            for i in range(cam_num):
                fg_pixel_indices = np.concatenate((fg_info['virtual_pixel_indices'][i][:,:3], fg_info['real_pixel_indices'][i][:,:3]), axis=0)
                if fg_info['virtual_points'][i].shape[1] == 3:
                    fg_info['virtual_points'][i] = np.concatenate((fg_info['virtual_points'][i], fg_info['virtual_pixel_indices'][i][:,-11:]), axis=1)
                    fg_info['real_points'][i] = np.concatenate((fg_info['real_points'][i], fg_info['real_pixel_indices'][i][:,-11:]), axis=1)
                fg_points_set = np.concatenate((fg_info['virtual_points'][i], fg_info['real_points'][i]), axis=0)
                timestamp = np.zeros((fg_points_set.shape[0], 1))
                fg_points_set = np.concatenate((fg_points_set, timestamp), axis=1)
                fg_pixels.append(fg_pixel_indices)
                fg_points.append(fg_points_set)
                fg_real_points_set = fg_info['real_points'][i]
                timestamp_real = np.zeros((fg_real_points_set.shape[0], 1))
                fg_real_points_set = np.concatenate((fg_real_points_set, timestamp_real), axis=1)
                fg_real_pixels.append(fg_info['real_pixel_indices'][i][:,:3])
                fg_real_points.append(fg_real_points_set)
            return dict(fg_pixels=fg_pixels, fg_points=fg_points, fg_real_pixels=fg_real_pixels, fg_real_points=fg_real_points)
        elif self.dataset == 'KittiDataset':
            if len(fg_info.keys()) == 4:
                fg_pixels = np.concatenate((fg_info['virtual_pixel_indices'], fg_info['real_pixel_indices']), axis=0)
                fg_points = np.concatenate((fg_info['virtual_points'], fg_info['real_points']), axis=0)
            else:
                fg_pixels = np.zeros((0,2))
                fg_points = np.zeros((0,6))
            return dict(fg_pixels=[fg_pixels], fg_points=[fg_points])
    def _make_point_class(self, fg_info):
        fg_points = fg_info['fg_points']
        cam_num = len(fg_points)
        point_class = get_points_type('LIDAR')
        for i in range(cam_num):
            fg_point = fg_points[i]
            fg_point = point_class(fg_point, points_dim=fg_point.shape[-1])
            fg_points[i] = fg_point
        fg_info['fg_points'] = fg_points
        return fg_info
    def __call__(self, results):
        if self.dataset == 'NuScenesDataset':
            pts_filename = results['pts_filename']
            tokens = pts_filename.split('/')
            try:
                fg_path = os.path.join(*tokens[:-2], "FOREGROUND_MIXED_6NN_WITH_DEPTH", tokens[-1]+'.pkl.npy')
                fg_info = np.load(fg_path, allow_pickle=True).item()
                fg_info = self._organize(fg_info)
                results["foreground2D_info"] = fg_info
            except FileNotFoundError: pass
            return results
        elif self.dataset == 'KittiDataset':
            pts_filename = results['pts_filename']
            tokens = pts_filename.split('/')
            fg_path = os.path.join(*tokens[:-2], "virtual_1NN", tokens[-1].split('.')[0]+'.npy')
            if os.path.exists(fg_path):
                fg_info = np.load(fg_path, allow_pickle=True).item()
                fg_info = self._organize(fg_info)
                fg_info = self._make_point_class(fg_info)
                results['foreground2D_info'] = fg_info
            return results
        else:
            raise NotImplementedError("foreground2D info of {} dataset is unavailable!".format(self.dataset))

@PIPELINES.register_module()
class LoadForeground2DFromMultiSweeps(object):
    def __init__(self, dataset="NuScenesDataset", sweeps_num=10):
        self.dataset = dataset
        self.sweeps_num = sweeps_num
    def _organize(self, fg_info, results, sweep):
        cam_num = len(fg_info['virtual_pixel_indices'])
        fg_pixels, fg_points, fg_real_pixels, fg_real_points = [], [], [], []
        ts = results['timestamp']
        sweep_ts = sweep['timestamp'] / 1e6
        for i in range(cam_num):
            fg_pixel_indices = np.concatenate((fg_info['virtual_pixel_indices'][i][:,:3], fg_info['real_pixel_indices'][i][:,:3]), axis=0)
            if fg_info['virtual_points'][i].shape[1] == 3:
                fg_info['virtual_points'][i] = np.concatenate((fg_info['virtual_points'][i], fg_info['virtual_pixel_indices'][i][:,-11:]), axis=1)
                fg_info['real_points'][i] = np.concatenate((fg_info['real_points'][i], fg_info['real_pixel_indices'][i][:,-11:]), axis=1)
            fg_points_set = np.concatenate((fg_info['virtual_points'][i], fg_info['real_points'][i]), axis=0)
            timestamp = np.zeros((fg_points_set.shape[0], 1))
            fg_points_set = np.concatenate((fg_points_set, timestamp), axis=1)
            fg_points_set[:, -1] = ts - sweep_ts
            fg_pixels.append(fg_pixel_indices)
            fg_points.append(fg_points_set)
            fg_real_points_set = fg_info['real_points'][i]
            timestamp_real = np.zeros((fg_real_points_set.shape[0], 1))
            fg_real_points_set = np.concatenate((fg_real_points_set, timestamp_real), axis=1)
            fg_real_points_set[:, -1] = ts - sweep_ts / 1e-6
            fg_real_pixels.append(fg_info['real_pixel_indices'][i][:,:3])
            fg_real_points.append(fg_real_points_set)
        return dict(fg_pixels=fg_pixels, fg_points=fg_points, fg_real_pixels=fg_real_pixels, fg_real_points=fg_real_points)
    def _merge_sweeps(self, fg_info, sweep_fg_info, sweep):
        fg_pixels, fg_points = fg_info['fg_pixels'], fg_info['fg_points']
        fg_real_pixels, fg_real_points = fg_info['fg_real_pixels'], fg_info['fg_real_points']
        sweep_fg_pixels, sweep_fg_points = sweep_fg_info['fg_pixels'], sweep_fg_info['fg_points']
        sweep_fg_real_pixels, sweep_fg_real_points = sweep_fg_info['fg_real_pixels'], sweep_fg_info['fg_real_points']
        if len(sweep_fg_points) == len(fg_points):
            cam_num = len(fg_pixels)
            for cam_id in range(cam_num):
                fg_pixels[cam_id] = np.concatenate((fg_pixels[cam_id], sweep_fg_pixels[cam_id]), axis=0)
                fg_point, sweep_fg_point = fg_points[cam_id], sweep_fg_points[cam_id]
                sweep_fg_point[:,:3] = sweep_fg_point[:,:3] @ sweep['sensor2lidar_rotation'].T
                sweep_fg_point[:,:3] = sweep_fg_point[:,:3] + sweep['sensor2lidar_translation']
                fg_points[cam_id] = np.concatenate((fg_point, sweep_fg_point), axis=0)
                fg_real_pixels[cam_id] = np.concatenate([fg_real_pixels[cam_id], sweep_fg_real_pixels[cam_id]], axis=0)
                fg_real_point, sweep_fg_real_point = fg_real_points[cam_id], sweep_fg_real_points[cam_id]
                sweep_fg_real_point[:,:3] = sweep_fg_real_point[:,:3] @ sweep['sensor2lidar_rotation'].T
                sweep_fg_real_point[:,:3] = sweep_fg_real_point[:,:3] + sweep['sensor2lidar_translation']
                fg_real_points[cam_id] = np.concatenate([fg_real_point, sweep_fg_real_point], axis=0)
        else: pass
        fg_info['fg_pixels'], fg_info['fg_points'] = fg_pixels, fg_points
        fg_info['fg_real_pixels'], fg_info['fg_real_points'] = fg_real_pixels, fg_real_points
        return fg_info
    def _make_point_class(self, fg_info):
        fg_points = fg_info['fg_points']
        cam_num = len(fg_points)
        point_class = get_points_type('LIDAR')
        for i in range(cam_num):
            fg_point = fg_points[i]
            fg_point = point_class(fg_point, points_dim=fg_point.shape[-1])
            fg_points[i] = fg_point
        fg_info['fg_points'] = fg_points
        return fg_info
    def __call__(self, results):
        if self.dataset == "NuScenesDataset":
            if "foreground2D_info" not in results: return results
            fg_info = results["foreground2D_info"]
            if len(results['sweeps']) <= self.sweeps_num: choices = np.arange(len(results['sweeps']))
            elif self.test_mode: choices = np.arange(self.sweeps_num)
            else: choices = np.random.choice(len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                pts_filename = sweep['data_path']
                tokens = pts_filename.split('/')
                sweep_fg_path = os.path.join(*tokens[:-2], "FOREGROUND_MIXED_6NN_WITH_DEPTH", tokens[-1]+'.pkl.npy')
                if os.path.exists(sweep_fg_path):
                    try:
                        sweep_fg_info = np.load(sweep_fg_path, allow_pickle=True).item()
                        sweep_fg_info = self._organize(sweep_fg_info, results, sweep)
                        fg_info = self._merge_sweeps(fg_info, sweep_fg_info, sweep)
                    except: continue
                else: continue
            fg_info = self._make_point_class(fg_info)
            results['foreground2D_info'] = fg_info
            return results

@PIPELINES.register_module()
class PointSegClassMapping(object):
    def __init__(self, valid_cat_ids, max_cat_id=40):
        assert max_cat_id >= np.max(valid_cat_ids), 'max_cat_id should be greater than maximum id in valid_cat_ids'
        self.valid_cat_ids = valid_cat_ids
        self.max_cat_id = int(max_cat_id)
        neg_cls = len(valid_cat_ids)
        self.cat_id2class = np.ones(self.max_cat_id + 1, dtype=np.int) * neg_cls
        for cls_idx, cat_id in enumerate(valid_cat_ids):
            self.cat_id2class[cat_id] = cls_idx
    def __call__(self, results):
        assert 'pts_semantic_mask' in results
        pts_semantic_mask = results['pts_semantic_mask']
        converted_pts_sem_mask = self.cat_id2class[pts_semantic_mask]
        results['pts_semantic_mask'] = converted_pts_sem_mask
        return results
    def __repr__(self):
        return f"{self.__class__.__name__}(valid_cat_ids={self.valid_cat_ids}, max_cat_id={self.max_cat_id})"

@PIPELINES.register_module()
class NormalizePointsColor(object):
    def __init__(self, color_mean):
        self.color_mean = color_mean
    def __call__(self, results):
        points = results['points']
        assert points.attribute_dims is not None and 'color' in points.attribute_dims.keys(), 'Expect points have color attribute'
        if self.color_mean is not None:
            points.color = points.color - points.color.new_tensor(self.color_mean)
        points.color = points.color / 255.0
        results['points'] = points
        return results
    def __repr__(self):
        return f"{self.__class__.__name__}(color_mean={self.color_mean})"

def paint2seg_func():
    detection_mapping = {'car': 0, 'truck': 1, 'construction_vehicle': 2, 'bus': 3, 'trailer': 4, 'barrier': 5, 'motorcycle': 6, 'bicycle': 7, 'pedestrian': 8, 'traffic_cone': 9, 'others': 10}
    class_2d_mapping = {'others': 0, 'car': 1, 'truck': 2, 'trailer': 3, 'bus': 4, 'construction_vehicle': 5, 'bicycle': 6, 'motorcycle': 7, 'pedestrian': 8, 'traffic_cone': 9, 'barrier': 10}
    paint2det = {class_2d_mapping[name]: detection_mapping[name] for name in class_2d_mapping}
    paint2seg_func = np.vectorize(paint2det.get)
    return paint2seg_func

@PIPELINES.register_module()
class LoadPointsFromFile(object):
    def __init__(self, coord_type, load_dim=6, use_dim=[0, 1, 2], tanh_dim=None, shift_height=False, use_color=False, painting=False, file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int): use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']
        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.tanh_dim = tanh_dim
        self.file_client_args = file_client_args.copy()
        # [自动纠错] disk -> local
        if self.file_client_args.get('backend') == 'disk':
            self.file_client_args['backend'] = 'local'
        self.detection_mapping = {'car': 0, 'truck': 1, 'construction_vehicle': 2, 'bus': 3, 'trailer': 4, 'barrier': 5, 'motorcycle': 6, 'bicycle': 7, 'pedestrian': 8, 'traffic_cone': 9, 'others': 10}
        self.painting = painting
        if self.painting: self.predict_fun = paint2seg_func()

    def _load_points(self, pts_filename):
        try:
            pts_bytes = mmcv.load(pts_filename, backend_args=self.file_client_args)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except Exception:
            if pts_filename.endswith('.npy'): points = np.load(pts_filename)
            else: points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def get_each_pc_gt_info(self, obj_points, class_name):
        assert class_name in list(self.detection_mapping.keys()),"class name not exist!"
        labels_2d = np.zeros((obj_points.shape[0], 11), np.float32)
        labels_2d[:, self.detection_mapping[class_name]] = 1
        return np.concatenate([obj_points, labels_2d], axis=1)

    def _load_painting(self, points, pts_filename, instance_name=None):
        if self.painting=='gt_aug':
            points = self.get_each_pc_gt_info(points, instance_name)
        else:
            painting_path = pts_filename.split('/')
            painting_path[-2] = 'LIDAR_TOP_MASK'
            painting_path = '/'.join(i for i in painting_path)
            try:
                painting_bytes = mmcv.load(painting_path, backend_args=self.file_client_args)
                predict_idx = np.frombuffer(painting_bytes, dtype='uint8')
            except Exception:
                predict_idx = np.fromfile(painting_path, dtype='uint8')
            predict_idx_2d = self.predict_fun(predict_idx)
            sample_predict_label = np.eye(11, dtype='uint8')[predict_idx_2d]
            points = np.concatenate([points, sample_predict_label], axis=1)
        return points

    def __call__(self, results, instance_name=None):
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None
        if self.tanh_dim is not None:
            assert isinstance(self.tanh_dim, list)
            assert max(self.tanh_dim) < points.shape[1]
            assert min(self.tanh_dim) > 2
            points[:, self.tanh_dim] = np.tanh(points[:, self.tanh_dim])
        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate([points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)
        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None: attribute_dims = dict()
            attribute_dims.update(dict(color=[points.shape[1] - 3, points.shape[1] - 2, points.shape[1] - 1,]))
        if self.painting:
            points = self._load_painting(points, pts_filename, instance_name)
        points_class = get_points_type(self.coord_type)
        points = points_class(points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points
        return results
    def __repr__(self):
        return f"{self.__class__.__name__}(shift_height={self.shift_height}, use_color={self.use_color}, file_client_args={self.file_client_args}, load_dim={self.load_dim}, use_dim={self.use_dim})"

@PIPELINES.register_module()
class LoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D."""

    def __init__(self,
                 with_bbox_3d=True,
                 with_label_3d=True,
                 with_attr_label=False,
                 with_mask_3d=False,
                 with_seg_3d=False,
                 with_bbox=False,
                 with_label=False,
                 with_mask=False,
                 with_seg=False,
                 with_bbox_depth=False,
                 poly2mask=True,
                 seg_3d_dtype='int',
                 file_client_args=dict(backend='disk')):
        
        super().__init__(
            with_bbox=with_bbox,
            with_label=with_label,
            with_mask=with_mask,
            with_seg=with_seg,
            poly2mask=poly2mask)

        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.seg_3d_dtype = seg_3d_dtype

    def _load_bboxes_3d(self, results):
        results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d']
        results['bbox3d_fields'].append('gt_bboxes_3d')
        return results

    def _load_bboxes_depth(self, results):
        results['centers2d'] = results['ann_info']['centers2d']
        results['depths'] = results['ann_info']['depths']
        return results

    def _load_labels_3d(self, results):
        results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
        return results

    def _load_attr_labels(self, results):
        results['attr_labels'] = results['ann_info']['attr_labels']
        return results

    def _load_masks_3d(self, results):
        pts_instance_mask_path = results['ann_info']['pts_instance_mask_path']

        try:
            mask_bytes = mmcv.load(pts_instance_mask_path, backend_args=self.backend_args)
            pts_instance_mask = np.frombuffer(mask_bytes, dtype=np.int)
        except Exception:
            pts_instance_mask = np.fromfile(
                pts_instance_mask_path, dtype=np.int64)

        results['pts_instance_mask'] = pts_instance_mask
        results['pts_mask_fields'].append('pts_instance_mask')
        return results

    def _load_semantic_seg_3d(self, results):
        pts_semantic_mask_path = results['ann_info']['pts_semantic_mask_path']

        try:
            mask_bytes = mmcv.load(pts_semantic_mask_path, backend_args=self.backend_args)
            pts_semantic_mask = np.frombuffer(
                mask_bytes, dtype=self.seg_3d_dtype).copy()
        except Exception:
            pts_semantic_mask = np.fromfile(
                pts_semantic_mask_path, dtype=np.int64)

        results['pts_semantic_mask'] = pts_semantic_mask
        results['pts_seg_fields'].append('pts_semantic_mask')
        return results

    def __call__(self, results):
        # 调用父类 (LoadAnnotations) 加载 2D 信息 (如 with_bbox=True)
        results = super().__call__(results)
        
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)
        if self.with_mask_3d:
            results = self._load_masks_3d(results)
        if self.with_seg_3d:
            results = self._load_semantic_seg_3d(results)

        return results

    def __repr__(self):
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_attr_label={self.with_attr_label}, '
        repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
        repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
        repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
        repr_str += f'{indent_str}with_label={self.with_label}, '
        repr_str += f'{indent_str}with_mask={self.with_mask}, '
        repr_str += f'{indent_str}with_seg={self.with_seg}, '
        repr_str += f'{indent_str}with_bbox_depth={self.with_bbox_depth}, '
        repr_str += f'{indent_str}poly2mask={self.poly2mask})'
        return repr_str
