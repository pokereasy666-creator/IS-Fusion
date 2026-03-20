# ----------------- 物理修复：针对自定义 Runner 基类迁移 -----------------
from mmdet3d.compat import Registry

try:
    from mmcv.runner import EpochBasedRunner
except ImportError:
    try:
        from mmengine.runner import EpochBasedRunner
    except ImportError:
        try:
            from mmengine.runner import Runner as EpochBasedRunner
        except ImportError:
            class EpochBasedRunner:
                def __init__(self, *args, **kwargs): pass

try:
    from mmcv.runner import RUNNERS
except ImportError:
    RUNNERS = Registry('runners')

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
