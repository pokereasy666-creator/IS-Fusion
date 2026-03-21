# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class EmptyCacheHook(Hook):
    """Call ``torch.cuda.empty_cache()`` to release GPU memory.

    Args:
        before_epoch (bool): Whether to call before each epoch.
            Defaults to False.
        after_epoch (bool): Whether to call after each epoch.
            Defaults to True.
        after_iter (bool): Whether to call after each iteration.
            Defaults to False.
    """

    def __init__(self,
                 before_epoch: bool = False,
                 after_epoch: bool = True,
                 after_iter: bool = False,
                 **kwargs):
        self._before_epoch = before_epoch
        self._after_epoch = after_epoch
        self._after_iter = after_iter

    def _empty_cache(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def before_train_epoch(self, runner) -> None:
        if self._before_epoch:
            self._empty_cache()

    def after_train_epoch(self, runner) -> None:
        if self._after_epoch:
            self._empty_cache()

    def after_train_iter(self, runner, batch_idx, data_batch=None,
                         outputs=None) -> None:
        if self._after_iter:
            self._empty_cache()

    def after_val_epoch(self, runner, metrics=None) -> None:
        if self._after_epoch:
            self._empty_cache()

    def after_val_iter(self, runner, batch_idx, data_batch=None,
                       outputs=None) -> None:
        if self._after_iter:
            self._empty_cache()

    def after_test_epoch(self, runner, metrics=None) -> None:
        if self._after_epoch:
            self._empty_cache()

    def after_test_iter(self, runner, batch_idx, data_batch=None,
                        outputs=None) -> None:
        if self._after_iter:
            self._empty_cache()
