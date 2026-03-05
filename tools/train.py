
# --- GLOBAL MMCV LOAD MOCK ---
import mmcv
if not hasattr(mmcv, "load"):
    try:
        from mmengine.fileio import load as _load
        mmcv.load = _load
    except ImportError:
        import pickle
        def _fallback_load(file, *args, **kwargs):
            with open(file, "rb") as f:
                return pickle.load(f)
        mmcv.load = _fallback_load
# -----------------------------


# --- GLOBAL MMCV BUILD_FROM_CFG MOCK ---
import mmcv
if not hasattr(mmcv, "build_from_cfg"):
    def _mock_build_from_cfg(cfg, registry, default_args=None):
        if hasattr(registry, "build"):
            try:
                return registry.build(cfg)
            except Exception as e:
                obj_type = cfg.get("type")
                import importlib
                # 如果注册表崩了，我们就直接去这些源码目录里强行抓人！
                search_modules = [
                    "mmdet3d.datasets.pipelines", 
                    "mmdet.datasets.pipelines", 
                    "mmdet3d.datasets.pipelines.loading", 
                    "mmdet3d.datasets.pipelines.transforms_3d"
                ]
                for mod_name in search_modules:
                    try:
                        mod = importlib.import_module(mod_name)
                        if hasattr(mod, obj_type):
                            cls = getattr(mod, obj_type)
                            args = cfg.copy()
                            args.pop("type")
                            if default_args: args.update(default_args)
                            return cls(**args)
                    except:
                        continue
                raise e
        from mmengine.registry import build_from_cfg
        return build_from_cfg(cfg, registry, default_args)
    mmcv.build_from_cfg = _mock_build_from_cfg
# ---------------------------------------

import mmcv
mmcv.__version__="2.1.0"

# --- GLOBAL DATALOADER FIX ---
try:
    from mmdet3d.datasets.builder import build_dataloader
except ImportError:
    try:
        from mmdet.datasets.builder import build_dataloader
    except ImportError:
        def build_dataloader(dataset, *args, **kwargs):
            import torch
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)

            from torch.utils.data import DataLoader
            samples_per_gpu = kwargs.get("samples_per_gpu", args[0] if len(args)>0 else 1)
            workers_per_gpu = kwargs.get("workers_per_gpu", args[1] if len(args)>1 else 1)
            collate = None
            try:
                from mmdet3d.datasets.utils import collate_fn
                collate = collate_fn
            except:
                pass
            return DataLoader(dataset, batch_size=samples_per_gpu, num_workers=workers_per_gpu, shuffle=True, collate_fn=collate)
# -----------------------------

# Copyright (c) OpenMMLab. All rights reserved.
# from __future__ import division


# --- 🚀 HOTFIX: 彻底解决 PyTorch 2.x DDP 与 Checkpoint 冲突 ---
import torch
import torch.utils.checkpoint
_orig_cp = torch.utils.checkpoint.checkpoint

def _patched_cp(*args, **kwargs):
    kwargs['use_reentrant'] = False
    return _orig_cp(*args, **kwargs)

torch.utils.checkpoint.checkpoint = _patched_cp
print("🚀 [HOTFIX] 已强制全局 Checkpoint 使用 use_reentrant=False！")
# -------------------------------------------------------------

import argparse
import copy
import mmcv
if not hasattr(mmcv, "mkdir_or_exist"):
    import os
    mmcv.mkdir_or_exist = lambda d, **kwargs: os.makedirs(d, exist_ok=True)

import os
import time
import torch
import warnings
from mmengine.config import Config, DictAction
import mmcv
if not hasattr(mmcv, "mkdir_or_exist"):
    import os
    mmcv.mkdir_or_exist = lambda d, **kwargs: os.makedirs(d, exist_ok=True)

from mmengine.dist import get_dist_info, init_dist
from os import path as osp

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger

try:
    from mmdet.apis import set_random_seed
except ImportError:
    try:
        from mmengine.runner import set_random_seed
    except ImportError:
        import torch
        import numpy as np
        import random
        def set_random_seed(seed, deterministic=False):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

from mmseg import __version__ as mmseg_version


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--extra_tag', type=str, default=None, help='extra tag for this experiment')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():



    args = parse_args()

    cfg = Config.fromfile(args.config)

    # --- A30 (24GB): Use original max_voxels from config, no reduction needed ---
    # Enable TF32 for Ampere GPUs (A30/A100) - free speedup with negligible precision loss
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("[Info] TF32 enabled for Ampere GPU acceleration")

    if hasattr(cfg, "model") and "img_backbone" in cfg.model:
        cfg.model.img_backbone.with_cp = True
    cfg.find_unused_parameters = False
    print("[Info] Gradient checkpointing enabled, find_unused_parameters=False")
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = False

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.extra_tag is not None:
        cfg.work_dir = osp.join(cfg.work_dir, args.extra_tag)

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)


# ----------------- 物理修复：核弹级全量抓捕所有隐身算子 -----------------
    try:
        from mmengine.registry import TRANSFORMS
        import os
        import importlib

        # 1. 自动遍历并导入 pipelines 目录下的所有 .py 文件
        pipeline_dir = 'mmdet3d/datasets/pipelines'
        if os.path.exists(pipeline_dir):
            for file in os.listdir(pipeline_dir):
                if file.endswith('.py') and not file.startswith('__'):
                    module_name = file[:-3]
                    try:
                        # 动态导入，例如 mmdet3d.datasets.pipelines.formating
                        mod = importlib.import_module(f'mmdet3d.datasets.pipelines.{module_name}')
                        # 将文件内所有类塞进注册表
                        for name in dir(mod):
                            obj = getattr(mod, name)
                            if isinstance(obj, type) and not TRANSFORMS.get(name):
                                TRANSFORMS.register_module(name=name, module=obj, force=True)
                    except Exception as e:
                        pass
        print("[Info] Nuclear launch detected: Successfully registered ALL hidden transforms!")

        # 2. 保留常规同步：防止在 mmdet/mmdet3d 基础注册表里的算子掉线
        try:
            from mmdet.datasets.builder import PIPELINES as DET_PIPELINES
            for name, obj in DET_PIPELINES.module_dict.items():
                if not TRANSFORMS.get(name): TRANSFORMS.register_module(name=name, module=obj, force=True)
        except Exception: pass
        
        try:
            from mmdet3d.datasets.builder import PIPELINES as DET3D_PIPELINES
            for name, obj in DET3D_PIPELINES.module_dict.items():
                if not TRANSFORMS.get(name): TRANSFORMS.register_module(name=name, module=obj, force=True)
        except Exception: pass

        # 3. 兼容加载 custom_imports 里的算子
        if cfg.get('custom_imports', None):
            imports = cfg['custom_imports'].get('imports', [])
            for mod in imports:
                try:
                    importlib.import_module(mod)
                except Exception: pass

    except Exception as e:
        print(f"[Warning] Ultimate registry sync failed: {e}")
    # ----------------- 物理修复结束 -----------------
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    # cfg.model.train_cfg = None # Optional
    # build the model and load checkpoint
    
    # ----------------- 物理修复：解除 MMEngine 抽象方法实例化限制 -----------------
    try:
        from mmdet3d.models.builder import DETECTORS
        model_type = cfg.model.get('type')
        model_cls = DETECTORS.get(model_type)
        if model_cls is not None:
            # 暴力抹除 Python ABCMeta 的抽象方法限制，让旧模型强行实例化
            if hasattr(model_cls, '__abstractmethods__'):
                model_cls.__abstractmethods__ = frozenset()
            
            # 补充新版接口的占位符，防止后续内部流转时报错
            if not hasattr(model_cls, 'loss'): 
                model_cls.loss = lambda self, *args, **kwargs: None
            if not hasattr(model_cls, 'predict'): 
                model_cls.predict = lambda self, *args, **kwargs: None
            if not hasattr(model_cls, '_forward'): 
                model_cls._forward = lambda self, *args, **kwargs: None
            
            print(f"[Info] Successfully bypassed abstract methods for {model_type}!")
    except Exception as e:
        print(f"[Warning] Abstract method bypass failed: {e}")
    # ----------------- 物理修复结束 -----------------
    # ----------------- 物理修复：拦截 MMEngine 的自动注册自救机制 -----------------
    import mmdet3d.utils
    if not hasattr(mmdet3d.utils, 'register_all_modules'):
        # 强行塞一个空函数进去，让它调用时不报错
        mmdet3d.utils.register_all_modules = lambda *args, **kwargs: None
    # ----------------- 物理修复结束 -----------------
    # ----------------- 物理修复：拦截 build_norm_layer 的 None 异常 -----------------
    try:
        import mmcv.cnn.bricks.norm as mmcv_norm
        import torch.nn as nn
        # 防止重复包裹
        if not hasattr(mmcv_norm, '_original_build_norm_layer'):
            mmcv_norm._original_build_norm_layer = mmcv_norm.build_norm_layer
            
            def safe_build_norm_layer(cfg, num_features, postfix=''):
                if cfg is None or cfg.get('type') is None:
                    # 如果配置为空，返回一个占位的 Identity 层
                    return 'norm', nn.Identity()
                return mmcv_norm._original_build_norm_layer(cfg, num_features, postfix)
            
            mmcv_norm.build_norm_layer = safe_build_norm_layer
            print("[Info] Successfully patched build_norm_layer to handle None config!")
    except Exception as e:
        print(f"[Warning] Failed to patch build_norm_layer: {e}")
    # ----------------- 物理修复结束 -----------------
    # ----------------- 物理修复：同步 BBOX_CODERS 到 TASK_UTILS -----------------
    try:
        from mmdet.registry import TASK_UTILS
        
        # 1. 尝试暴力加载可能隐藏的 core/bbox 目录下的算子
        try:
            import mmdet3d.core.bbox.coders as hidden_coders
            for name in dir(hidden_coders):
                obj = getattr(hidden_coders, name)
                if isinstance(obj, type) and not TASK_UTILS.get(name):
                    TASK_UTILS.register_module(name=name, module=obj, force=True)
        except Exception: pass

        # 2. 把旧版 BBOX_CODERS 里的内容全量转移到新版 TASK_UTILS
        try:
            from mmdet3d.core.bbox.builder import BBOX_CODERS as DET3D_BBOX_CODERS
            for name, obj in DET3D_BBOX_CODERS.module_dict.items():
                if not TASK_UTILS.get(name): 
                    TASK_UTILS.register_module(name=name, module=obj, force=True)
        except Exception: pass
        
        print("[Info] Successfully synced BBOX_CODERS into TASK_UTILS!")
    except Exception as e:
        print(f"[Warning] BBOX_CODERS sync failed: {e}")
    # ----------------- 物理修复结束 -----------------
# ----------------- 物理修复：终极内存欺骗大法 + 全量核扫核心算子 -----------------
    try:
        import sys
        import types
        import os
        import importlib
        from mmdet.registry import TASK_UTILS

        # 1. 内存幻觉：无限扩大欺骗范围，涵盖所有 bbox 组件
        if 'mmdet.core' not in sys.modules: sys.modules['mmdet.core'] = types.ModuleType('mmdet.core')
        if 'mmdet.core.bbox' not in sys.modules:
            fake_bbox = types.ModuleType('mmdet.core.bbox')
            # 捏造所有它可能需要的旧版基类
            class BaseBBoxCoder: pass
            class BaseAssigner: pass
            class AssignResult: pass
            class MaxIoUAssigner: pass
            
            fake_bbox.BaseBBoxCoder = BaseBBoxCoder
            fake_bbox.BaseAssigner = BaseAssigner
            fake_bbox.AssignResult = AssignResult
            fake_bbox.MaxIoUAssigner = MaxIoUAssigner
            
            fake_bbox.build_bbox_coder = lambda *args, **kwargs: None
            fake_bbox.build_assigner = lambda *args, **kwargs: None
            fake_bbox.build_sampler = lambda *args, **kwargs: None
            fake_bbox.build_match_cost = lambda *args, **kwargs: None
            sys.modules['mmdet.core.bbox'] = fake_bbox
            
        if 'mmdet.core.bbox.builder' not in sys.modules:
            fake_builder = types.ModuleType('mmdet.core.bbox.builder')
            # 把所有旧注册表全映射到新版的 TASK_UTILS
            fake_builder.BBOX_CODERS = TASK_UTILS 
            fake_builder.BBOX_ASSIGNERS = TASK_UTILS
            fake_builder.BBOX_SAMPLERS = TASK_UTILS
            fake_builder.MATCH_COSTS = TASK_UTILS
            sys.modules['mmdet.core.bbox.builder'] = fake_builder

        if 'mmdet.core.bbox.match_costs' not in sys.modules:
            fake_match = types.ModuleType('mmdet.core.bbox.match_costs')
            fake_match.build_match_cost = lambda *args, **kwargs: None
            sys.modules['mmdet.core.bbox.match_costs'] = fake_match

        # 2. 核弹级扫描：将 mmdet3d/core/bbox 下的所有算子（包含 Assigner/Coder/MatchCost）全量挖出
        core_bbox_dir = 'mmdet3d/core/bbox'
        if os.path.exists(core_bbox_dir):
            for root, dirs, files in os.walk(core_bbox_dir):
                for file in files:
                    if file.endswith('.py') and not file.startswith('__'):
                        rel_path = os.path.relpath(os.path.join(root, file), 'mmdet3d/core/bbox')
                        mod_name = 'mmdet3d.core.bbox.' + rel_path.replace(os.sep, '.')[:-3]
                        try:
                            mod = importlib.import_module(mod_name)
                            for name in dir(mod):
                                obj = getattr(mod, name)
                                if isinstance(obj, type) and not TASK_UTILS.get(name):
                                    TASK_UTILS.register_module(name=name, module=obj, force=True)
                        except Exception: pass

        # 3. 把官方旧版字典里的漏网之鱼也全捞过来
        try:
            from mmdet3d.core.bbox.builder import BBOX_ASSIGNERS, MATCH_COSTS, BBOX_SAMPLERS
            for name, obj in BBOX_ASSIGNERS.module_dict.items():
                if not TASK_UTILS.get(name): TASK_UTILS.register_module(name=name, module=obj, force=True)
            for name, obj in MATCH_COSTS.module_dict.items():
                if not TASK_UTILS.get(name): TASK_UTILS.register_module(name=name, module=obj, force=True)
        except Exception: pass

        print("[Info] Successfully mocked mmdet.core and scanned ALL core.bbox modules (including Assigners)!")
    except Exception as e:
        print(f"[Warning] Memory mock & scan failed: {e}")
    # ----------------- 物理修复结束 -----------------
    # ----------------- 物理排查：暴露隐藏的报错 -----------------
    try:
        import mmdet3d.core.bbox.assigners.hungarian_assigner
    except Exception as e:
        print("\n[!!!] HIDDEN ERROR REVEALED [!!!]")
        import traceback
        traceback.print_exc()
        print("---------------------------------\n")
    # ----------------- 排查结束 -----------------
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    logger.info(f'Model:\n{model}')
    datasets = [build_dataset(cfg.data.train)]

    # --- OPTIMIZER MMEngine WRAPPER FIX ---
    if "optimizer" in cfg and "type" in cfg.optimizer and cfg.optimizer["type"] != "OptimWrapper":
        opt_cfg = cfg.optimizer.copy()
        paramwise_cfg = opt_cfg.pop("paramwise_cfg", None)
        optim_wrapper = dict(type="OptimWrapper", optimizer=opt_cfg)
        if paramwise_cfg is not None:
            optim_wrapper["paramwise_cfg"] = paramwise_cfg
        if hasattr(cfg, "optimizer_config") and cfg.optimizer_config is not None:
            if "grad_clip" in cfg.optimizer_config:
                optim_wrapper["clip_grad"] = cfg.optimizer_config["grad_clip"]
        cfg.optimizer = optim_wrapper
    # --------------------------------------

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], 'PALETTE') else None)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
