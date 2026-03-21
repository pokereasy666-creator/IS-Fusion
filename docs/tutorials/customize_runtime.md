# Tutorial 5: Customize Runtime Settings

## Customize optimization settings

### Customize optimizer supported by PyTorch

We already support to use all the optimizers implemented by PyTorch, and the only modification is to change the `optimizer` field of config files.
For example, if you want to use `ADAM` (note that the performance could drop a lot), the modification could be as the following.

```python
optimizer = dict(type='Adam', lr=0.0003, weight_decay=0.0001)
```

To modify the learning rate of the model, the users only need to modify the `lr` in the config of optimizer. The users can directly set arguments following the [API doc](https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim) of PyTorch.

### Customize self-implemented optimizer

#### 1. Define a new optimizer

A customized optimizer could be defined as following.

Assume you want to add a optimizer named `MyOptimizer`, which has arguments `a`, `b`, and `c`.
You need to create a new directory named `mmdet3d/core/optimizer`.
And then implement the new optimizer in a file, e.g., in `mmdet3d/core/optimizer/my_optimizer.py`:

```python
from mmengine.registry import OPTIMIZERS
from torch.optim import Optimizer


@OPTIMIZERS.register_module()
class MyOptimizer(Optimizer):

    def __init__(self, a, b, c)

```

#### 2. Add the optimizer to registry

To find the above module defined above, this module should be imported into the main namespace at first. There are two options to achieve it.

- Add `mmdet3d/core/optimizer/__init__.py` to import it.

    The newly defined module should be imported in `mmdet3d/core/optimizer/__init__.py` so that the registry will
    find the new module and add it:

```python
from .my_optimizer import MyOptimizer

__all__ = ['MyOptimizer']

```

You also need to import `optimizer` in `mmdet3d/core/__init__.py` by adding:

```python
from .optimizer import *
```

Or use `custom_imports` in the config to manually import it

```python
custom_imports = dict(imports=['mmdet3d.core.optimizer.my_optimizer'], allow_failed_imports=False)
```

The module `mmdet3d.core.optimizer.my_optimizer` will be imported at the beginning of the program and the class `MyOptimizer` is then automatically registered.
Note that only the package containing the class `MyOptimizer` should be imported.
`mmdet3d.core.optimizer.my_optimizer.MyOptimizer` **cannot** be imported directly.

Actually users can use a totally different file directory structure in this importing method, as long as the module root can be located in `PYTHONPATH`.

#### 3. Specify the optimizer in the config file

Then you can use `MyOptimizer` in `optimizer` field of config files.
In the configs, the optimizers are defined by the field `optimizer` like the following:

```python
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
```

To use your own optimizer, the field can be changed to

```python
optimizer = dict(type='MyOptimizer', a=a_value, b=b_value, c=c_value)
```

### Customize optimizer constructor

Some models may have some parameter-specific settings for optimization, e.g. weight decay for BatchNorm layers.
The users can tune those fine-grained parameters through customizing optimizer constructor.

```python
from mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS
from mmengine.optim import DefaultOptimWrapperConstructor
from .my_optimizer import MyOptimizer


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class MyOptimizerConstructor(DefaultOptimWrapperConstructor):

    def __init__(self, optim_wrapper_cfg, paramwise_cfg=None):

    def __call__(self, model):

        return my_optimizer

```

The default optimizer wrapper constructor is implemented in [mmengine](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/default_constructor.py), which could also serve as a template for new optimizer constructor.

### Additional settings

Tricks not implemented by the optimizer should be implemented through optimizer constructor (e.g., set parameter-wise learning rates) or hooks. We list some common settings that could stabilize the training or accelerate the training. Feel free to create PR, issue for more settings.

- __Use gradient clip to stabilize training__:

    Some models need gradient clip to clip the gradients to stabilize the training process. In OpenMMLab v2, gradient clipping is configured inside `optim_wrapper`. An example is as below:

    ```python
    optim_wrapper = dict(
        optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
        clip_grad=dict(max_norm=35, norm_type=2))
    ```

    If your config inherits the base config which already sets the `optim_wrapper`, you might need `_delete_=True` to override the unnecessary settings in the base config. See the [config documentation](https://mmdetection.readthedocs.io/en/latest/tutorials/config.html) for more details.

- __Use param_scheduler to accelerate model convergence__:

    We support parameter schedulers to modify the learning rate and momentum according to a schedule, which could make the model converge in a faster way.
    For example, the following config is used in 3D detection to accelerate convergence.
    For more details, please refer to the [MMEngine param_scheduler documentation](https://mmengine.readthedocs.io/en/latest/tutorials/param_scheduler.html).

    ```python
    param_scheduler = [
        dict(
            type='CosineAnnealingLR',
            T_max=8,
            eta_min=1e-5,
            begin=0,
            end=8),
    ]
    ```

## Customize training schedules

By default we use step learning rate with 1x schedule. In OpenMMLab v2, learning rate schedules are configured via `param_scheduler` using [MMEngine's scheduler classes](https://mmengine.readthedocs.io/en/latest/tutorials/param_scheduler.html). Here are some examples

- Poly schedule:

    ```python
    param_scheduler = [
        dict(type='PolyLR', power=0.9, eta_min=1e-4, by_epoch=False)
    ]
    ```

- CosineAnnealing schedule:

    ```python
    param_scheduler = [
        dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
        dict(type='CosineAnnealingLR', eta_min_ratio=1e-5)
    ]
    ```

## Customize training loop

In OpenMMLab v2, the training loop is configured via `train_cfg`, `val_cfg`, and `test_cfg`. By default, an epoch-based training loop is used:

```python
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict()
test_cfg = dict()
```

`val_interval` controls how often validation is run (every N training epochs). If you don't need validation, you can omit `val_cfg`.

For iteration-based training, use:

```python
train_cfg = dict(type='IterBasedTrainLoop', max_iters=90000, val_interval=5000)
```

## Customize hooks

### Customize self-implemented hooks

#### 1. Implement a new hook

There are some occasions when the users might need to implement a new hook. MMDetection supports customized hooks in training (#3395) since v2.3.0. Thus the users could implement a hook directly in mmdet or their mmdet-based codebases and use the hook by only modifying the config in training.
Before v2.3.0, the users need to modify the code to get the hook registered before training starts.
Here we give an example of creating a new hook in mmdet3d and using it in training.

```python
from mmengine.registry import HOOKS
from mmengine.hooks import Hook


@HOOKS.register_module()
class MyHook(Hook):

    def __init__(self, a, b):
        pass

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_train_epoch(self, runner):
        pass

    def after_train_epoch(self, runner):
        pass

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        pass

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        pass
```

Depending on the functionality of the hook, the users need to specify what the hook will do at each stage of the training in `before_run`, `after_run`, `before_train_epoch`, `after_train_epoch`, `before_train_iter`, and `after_train_iter`.

#### 2. Register the new hook

Then we need to make `MyHook` imported. Assuming the hook is in `mmdet3d/core/utils/my_hook.py` there are two ways to do that:

- Modify `mmdet3d/core/utils/__init__.py` to import it.

    The newly defined module should be imported in `mmdet3d/core/utils/__init__.py` so that the registry will
    find the new module and add it:

```python
from .my_hook import MyHook

__all__ = [..., 'MyHook']

```

Or use `custom_imports` in the config to manually import it

```python
custom_imports = dict(imports=['mmdet3d.core.utils.my_hook'], allow_failed_imports=False)
```

#### 3. Modify the config

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value)
]
```

You can also set the priority of the hook by setting key `priority` to `'NORMAL'` or `'HIGHEST'` as below

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value, priority='NORMAL')
]
```

By default the hook's priority is set as `NORMAL` during registration.

### Use hooks implemented in MMCV

If the hook is already implemented in MMCV, you can directly modify the config to use the hook as below

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value, priority='NORMAL')
]
```

### Modify default runtime hooks

In OpenMMLab v2, common runtime hooks are configured through `default_hooks` rather than individual top-level config fields. The default hooks include:

- `checkpoint` - for saving checkpoints
- `logger` - for logging
- `param_scheduler` - for learning rate and momentum scheduling
- `timer` - for timing
- `sampler_seed` - for setting random seeds

The above-mentioned tutorials already cover how to modify `optim_wrapper` and `param_scheduler`.
Here we reveal what we can do with checkpoint and logger configuration through `default_hooks`.

#### Checkpoint config

In v2, checkpoint saving is configured via `default_hooks`. See the [MMEngine documentation](https://mmengine.readthedocs.io/en/latest/tutorials/hook.html) for more details.

```python
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1))
```

The users could set `max_keep_ckpts` to save only a small number of checkpoints or decide whether to store the state dict of the optimizer by `save_optimizer`.

#### Log config

In v2, logging is configured via `default_hooks` and `visualizer`. MMEngine supports `WandbVisBackend`, `MLflowVisBackend`, and `TensorboardVisBackend`.
The detailed usages can be found in the [MMEngine docs](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/visualization.html).

```python
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50))
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ])
```

#### Evaluation config

In v2, evaluation is configured via `val_evaluator` and `val_cfg`. The `val_evaluator` specifies the metrics, and `val_cfg` controls the validation loop.

```python
val_evaluator = dict(type='IndoorMetric')
val_cfg = dict()
```
