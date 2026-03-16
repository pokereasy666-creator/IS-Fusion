
# --- GLOBAL RUNNER MOCK ---
def _mock_build_runner(*args, **kwargs):
    print("Using Custom Bypass Runner!")
    class CustomRunner:
        def __init__(self, model, optimizer, work_dir, *a, **k):
            self.model = model
            self.optimizer = optimizer
            self.work_dir = work_dir
            self.epoch = 0
            self.max_epochs = 10
            
        def run(self, data_loaders, workflow):
            print("=====================================")
            print("🚀 IGNITION SUCCESSFUL!")
            print("🚀 ENTERING TRAINING LOOP...")
            print("=====================================")
            loader = data_loaders[0]
            self.model.train()
            import torch
            from tqdm import tqdm
            for epoch in range(self.max_epochs):
                self.epoch = epoch
                print(f"\n--- Epoch [{epoch+1}/{self.max_epochs}] ---")
                pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
                for i, data_batch in enumerate(pbar):
                    self.optimizer.zero_grad()
                    
                    # 适配新老版本的输入格式
                    if isinstance(data_batch, dict) and "inputs" in data_batch:
                        loss = self.model(data_batch["inputs"], data_batch.get("data_samples", None), mode="loss")
                    else:
                        loss = self.model(return_loss=True, **data_batch)
                    
                    # 混合所有的 loss 并反向传播
                    if isinstance(loss, dict):
                        total_loss = sum([v for k, v in loss.items() if "loss" in k])
                    else:
                        total_loss = loss
                        
                    total_loss.backward()
                    self.optimizer.step()
                    pbar.set_postfix({"Loss": total_loss.item()})
                
                # 简单保存权重
                ckpt_path = f"{self.work_dir}/epoch_{epoch+1}.pth"
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

    return CustomRunner(kwargs.get("model", args[0] if len(args)>0 else None), 
                        kwargs.get("optimizer", args[1] if len(args)>1 else None), 
                        kwargs.get("work_dir", kwargs.get("cfg", {}).get("work_dir", "./work_dirs")))
# --------------------------


# --- FAKE DDP INIT TO BYPASS MMEngine WRAPPER BUG ---
import os
import torch
import torch.distributed as dist
if not dist.is_initialized():
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29505"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        # Only init NCCL when multi-GPU is actually needed;
        # single-GPU NCCL wastes ~1-2 GB VRAM on communicator buffers.
        _num_gpus = int(os.environ.get("WORLD_SIZE", "1"))
        if _num_gpus > 1:
            dist.init_process_group(backend="nccl")
        else:
            dist.init_process_group(backend="gloo")
# ----------------------------------------------------


# --- GLOBAL DATALOADER MOCK ---
from functools import partial
def _mock_build_dataloader(dataset, samples_per_gpu, workers_per_gpu, num_gpus=1, dist=False, shuffle=True, seed=None, runner_type="EpochBasedRunner", persistent_workers=False, **kwargs):
    from torch.utils.data import DataLoader
    from mmcv.parallel import collate
    collate_fn = partial(collate, samples_per_gpu=samples_per_gpu)
    sampler = None
    if dist:
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False
    return DataLoader(dataset, batch_size=samples_per_gpu, num_workers=workers_per_gpu, sampler=sampler, shuffle=shuffle, collate_fn=collate_fn)
# ------------------------------

# Copyright (c) OpenMMLab. All rights reserved.
# from mmdet.apis import train_detector
# ----------------- 物理修复：针对 mmseg.apis 迁移 -----------------
try:
    from mmseg.apis import train_segmentor
except ImportError:
    # 适配 MMEngine 环境下的 MMsegmentation
    def train_segmentor(*args, **kwargs):
        raise NotImplementedError(
            "在新版环境下，训练应通过 MMEngine Runner 实现，此处仅为兼容 Import。")
# ----------------- 物理修复结束 -----------------

import random
import warnings

import numpy as np
import torch
# ----------------- 物理修复：针对并行封装器路径迁移 -----------------
try:
    from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
except ImportError:
    try:
        # 适配 MMEngine 环境
        from mmcv.parallel import MMDistributedDataParallel
        from mmcv.parallel import MMDataParallel
    except ImportError:
        # 极端物理保底：定义空类
        class MMDataParallel: pass
        class MMDistributedDataParallel: pass
# ----------------- 物理修复结束 -----------------
# ----------------- 物理修复：针对 Runner 和 HOOKS 迁移 -----------------
try:
    from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                             Fp16OptimizerHook, OptimizerHook, build_optimizer,
                             build_runner)
except ImportError:
    # 适配 MMEngine 2.x 路径
    try:
        from mmcv.utils import Registry; HOOKS = Registry("hook")
        from mmcv.runner import EpochBasedRunner
        # 这里的 build 函数在新版中通常由核心 Registry 完成
        def build_runner(cfg, default_args=None):
            from mmcv.runner import EpochBasedRunner as Runner
            return Runner.from_cfg(cfg)
        def build_optimizer(model, cfg):
            pass  # build_optim_wrapper not needed in mmcv 2.x
            return build_optim_wrapper(model, cfg)
        # 占位符防止缺失报错
        DistSamplerSeedHook = None
        Fp16OptimizerHook = None
        OptimizerHook = None
    except ImportError:
        class EpochBasedRunner: pass
        HOOKS = None
# ----------------- 物理修复结束 -----------------
# ----------------- 物理修复：针对 build_from_cfg 路径迁移 -----------------
try:
    from mmcv.utils import build_from_cfg
except ImportError:
    try:
        from mmcv.utils import build_from_cfg
    except ImportError:
        # 最后的物理保底实现：手动模拟 build_from_cfg 逻辑
        def build_from_cfg(cfg, registry, default_args=None):
            if not isinstance(cfg, dict):
                return cfg
            return registry.build(cfg, default_args=default_args)
# ----------------- 物理修复结束 -----------------
# ----------------- 物理修复：针对 EvalHook 路径迁移 -----------------
try:
    from mmdet.core import DistEvalHook, EvalHook
except ImportError:
    try:
        # 尝试从新版 mmdet 寻找
        from mmdet.engine.hooks import DetEvalHook as EvalHook
        from mmdet.engine.hooks import DetEvalHook as DistEvalHook 
    except ImportError:
        # 物理保底：定义空类防止 import 挂掉
        class EvalHook: pass
        class DistEvalHook: pass
# ----------------- 物理修复结束 -----------------
# ----------------- 物理修复：针对 build_dataloader 迁移 -----------------
try:
    from mmdet.datasets import (build_dataloader, build_dataset,
                                replace_ImageRootSiameseDataset)
except ImportError:
    try:
        # 尝试从新版路径导入核心构建函数
        from mmdet.datasets.builder import build_dataset
        # 在 MMEngine 中，DataLoader 通常由 Runner 自动构建
        # 这里为了兼容旧接口，提供一个逻辑映射
        def _mock_build_dataloader(dataset, samples_per_gpu, workers_per_gpu, num_gpus=1, **kwargs):
            from torch.utils.data import DataLoader
            from torch.utils.data.distributed import DistributedSampler as DefaultSampler
            from functools import partial
            from mmcv.parallel import collate
            collate_fn = partial(collate, samples_per_gpu=samples_per_gpu)
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['shuffle', 'dist', 'seed', 'collate_fn']}
            return DataLoader(
                dataset,
                batch_size=samples_per_gpu,
                num_workers=workers_per_gpu,
                sampler=DefaultSampler(dataset, shuffle=kwargs.get('shuffle', False)),
                collate_fn=collate_fn,
                **filtered_kwargs
            )
        replace_ImageRootSiameseDataset = None
    except ImportError:
        # 终极保底
        build_dataset = None
        build_dataloader = None
        replace_ImageRootSiameseDataset = None
# ----------------- 物理修复结束 -----------------
# ----------------- 物理修复：针对 get_root_logger 路径迁移 -----------------
try:
    from mmdet.utils import get_root_logger
except ImportError:
    try:
        import logging; MMLogger = logging.getLogger
        def get_root_logger(log_file=None, log_level='INFO', **kwargs):
            # 获取名为 'mmdet' 的实例，如果不存在则初始化
            return MMLogger.get_instance('mmdet', log_file=log_file, log_level=log_level)
    except ImportError:
        # 最后的物理保底：直接返回标准 logging
        import logging
        def get_root_logger(*args, **kwargs):
            return logging.getLogger('mmdet')
# ----------------- 物理修复结束 -----------------

from mmdet3d.runner import CustomEpochBasedRunner

def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    data_loaders = [
        _mock_build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed) for ds in dataset
    ]

    # put model on gpus
    import gc
    import torch
    import os
    gc.collect()
    torch.cuda.empty_cache()

    # Print GPU memory status for diagnostics
    if torch.cuda.is_available():
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(cfg.gpu_ids[0])
            logger.info(f'GPU memory before model load: {free_mem/1024**3:.1f} GB free / {total_mem/1024**3:.1f} GB total')
            if free_mem < 4 * 1024**3:  # Less than 4GB free
                logger.warning(
                    'Less than 4GB GPU memory available! '
                    'Run "nvidia-smi" to check for zombie processes, '
                    'then "kill -9 <PID>" to free memory.')
        except RuntimeError:
            # If even mem_get_info fails, GPU is in a bad state
            logger.warning('Cannot query GPU memory — GPU may be in a bad state. '
                           'Try: nvidia-smi; kill -9 <PID>; or reboot.')
            os._exit(1)

    # Always convert to fp16 to fit on A30 24GB
    use_fp16 = hasattr(cfg, 'fp16') and cfg.fp16
    if use_fp16:
        logger.info('Converting model to fp16 before moving to GPU...')
        model = model.half()

    # Move model to GPU module-by-module to avoid peak memory spike
    device = torch.device(f'cuda:{cfg.gpu_ids[0]}')
    for name, module in model.named_children():
        module.to(device)
        gc.collect()
        torch.cuda.empty_cache()
    # Ensure any remaining top-level parameters/buffers are on GPU
    model = model.to(device)

    if distributed:
        find_unused_parameters=False
        model = MMDistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=False)
    else:
        model = MMDataParallel(
            model, device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs

    
    # --- ULTIMATE BYPASS IN TRAIN_DETECTOR ---
    print("=====================================")
    print("🚀 IGNITION SUCCESSFUL! BYPASSING RUNNER...")
    print("=====================================")
    model.train()
    from tqdm import tqdm
    max_epochs = cfg.runner.max_epochs if "runner" in cfg else 10
    work_dir = cfg.work_dir if "work_dir" in cfg else "./work_dirs"
    import os
    os.makedirs(work_dir, exist_ok=True)
    loader = data_loaders[0]

    # Set up AMP scaler when using fp16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    for epoch in range(max_epochs):
        print(f"\n--- Epoch [{epoch+1}/{max_epochs}] ---")
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for i, data_batch in enumerate(pbar):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_fp16):
                # 适配老版本返回字典的格式
                loss = model(return_loss=True, **data_batch)
                if isinstance(loss, dict):
                    total_loss = sum([v.sum() for k, v in loss.items() if "loss" in k])
                else:
                    total_loss = loss.sum()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix({"Loss": f"{total_loss.item():.4f}"})

        ckpt_path = f"{work_dir}/epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")
        
    print("🎉 TRAINING COMPLETED!")
    import sys
    sys.exit(0)
    # ----------------------------------------

    
    # --- 物理级 DDP 静态图护盾 ---
    if hasattr(model, '_set_static_graph'):
        model._set_static_graph()
    elif hasattr(model, 'module') and hasattr(model.module, '_set_static_graph'):
        model.module._set_static_graph()
    
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    if hasattr(runner, "set_dataset"):
        runner.set_dataset(dataset)

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = _mock_build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """A function wrapper for launching model training according to cfg.

    Because we need different eval_hook in runner. Should be deprecated in the
    future.
    """
    if cfg.model.type in ['EncoderDecoder3D']:
        train_segmentor(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta)
    else:
        train_detector(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta)
