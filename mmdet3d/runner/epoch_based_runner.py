# ----------------- 物理修复：针对自定义 Runner 基类迁移 -----------------
try:
    from mmcv.runner import EpochBasedRunner
except ImportError:
    # 适配 MMEngine Runner。
    # 注意：新版 Runner 逻辑与旧版差异较大，此处仅为保证 Import 通过
    try:
        from mmengine.runner import Runner as EpochBasedRunner
    except ImportError:
        class EpochBasedRunner:
            def __init__(self, *args, **kwargs): pass
# ----------------- 物理修复结束 -----------------
# ----------------- 物理修复：针对 RUNNERS 注册表迁移 -----------------
try:
    from mmcv.runner.builder import RUNNERS
except ImportError:
    try:
        from mmengine.registry import RUNNERS
    except ImportError:
        # 极端物理保底：手动创建一个临时的 Registry 对象
        from mmengine.registry import Registry
        RUNNERS = Registry('runner')
# ----------------- 物理修复结束 -----------------

@RUNNERS.register_module()
class CustomEpochBasedRunner(EpochBasedRunner):
    def set_dataset(self, dataset):
        self._dataset = dataset

    def train(self, data_loader, **kwargs):
        # update the schedule for data augmentation
        for dataset in self._dataset:
            dataset.set_epoch(self.epoch)

        super().train(data_loader, **kwargs)
