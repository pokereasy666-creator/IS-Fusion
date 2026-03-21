# 教程 5：自定义运行时配置

## 自定义优化器设置

### 自定义 PyTorch 支持的优化器

我们已经支持使用所有 PyTorch 实现的优化器，且唯一需要修改的地方就是改变配置文件中的 `optimizer` 字段。
举个例子，如果您想使用 `ADAM` （注意到这样可能会使性能大幅下降），您可以这样修改：

```python
optimizer = dict(type='Adam', lr=0.0003, weight_decay=0.0001)
```

为了修改模型的学习率，用户只需要修改优化器配置中的 `lr` 字段。用户可以根据 PyTorch 的 [API 文档](https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim) 直接设置参数。

### 自定义并实现优化器

#### 1. 定义新的优化器

一个自定义优化器可以按照如下过程定义：

假设您想要添加一个叫 `MyOptimizer` 的，拥有参数 `a`，`b` 和 `c` 的优化器，您需要创建一个叫做 `mmdet3d/core/optimizer` 的目录。
接下来，应该在目录下某个文件中实现新的优化器，比如 `mmdet3d/core/optimizer/my_optimizer.py`：

```python
from mmengine.registry import OPTIMIZERS
from torch.optim import Optimizer


@OPTIMIZERS.register_module()
class MyOptimizer(Optimizer):

    def __init__(self, a, b, c)

```

#### 2. 将优化器添加到注册器

为了找到上述定义的优化器模块，该模块首先需要被引入主命名空间。有两种方法实现之：

- 新建 `mmdet3d/core/optimizer/__init__.py` 文件用于引入。

    新定义的模块应该在 `mmdet3d/core/optimizer/__init__.py` 中被引入，使得注册器可以找到新模块并注册之：

```python
from .my_optimizer import MyOptimizer

__all__ = ['MyOptimizer']

```

您也需要通过添加如下语句在 `mmdet3d/core/__init__.py` 中引入 `optimizer`：

```python
from .optimizer import *
```

或者在配置中使用 `custom_imports` 来人工引入新优化器：

```python
custom_imports = dict(imports=['mmdet3d.core.optimizer.my_optimizer'], allow_failed_imports=False)
```

模块 `mmdet3d.core.optimizer.my_optimizer` 会在程序伊始被引入，且 `MyOptimizer` 类在那时会自动被注册。
注意到只有包含 `MyOptimizer` 类的包应该被引入。
`mmdet3d.core.optimizer.my_optimizer.MyOptimizer` **不能** 被直接引入。

事实上，用户可以在这种引入的方法中使用完全不同的文件目录结构，只要保证根目录能在 `PYTHONPATH` 中被定位。

#### 3. 在配置文件中指定优化器

接下来您可以在配置文件的 `optimizer` 字段中使用 `MyOptimizer`。
在配置文件中，优化器在 `optimizer` 字段中以如下方式定义：

```python
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
```

为了使用您自己的优化器，该字段可以改为：

```python
optimizer = dict(type='MyOptimizer', a=a_value, b=b_value, c=c_value)
```

### 自定义优化器的构造器

部分模型可能会拥有一些参数专属的优化器设置，比如 BatchNorm 层的权重衰减 (weight decay)。
用户可以通过自定义优化器的构造器来对那些细粒度的参数进行调优。

```python
from mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS, OPTIMIZERS
from mmengine.optim import DefaultOptimWrapperConstructor
from .my_optimizer import MyOptimizer


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class MyOptimizerConstructor(DefaultOptimWrapperConstructor):

    def __init__(self, optim_wrapper_cfg, paramwise_cfg=None):

    def __call__(self, model):

        return my_optimizer

```

默认优化器封装构造器在 [MMEngine](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/default_constructor.py) 中实现。这部分代码也可以用作新优化器构造器的模版。

### 额外的设置

没有在优化器部分实现的技巧应该通过优化器构造器或者钩子来实现 （比如逐参数的学习率设置）。我们列举了一些常用的可以稳定训练过程或者加速训练的设置。我们欢迎提供更多类似设置的 PR 和 issue。

- __使用梯度裁剪 (gradient clip) 来稳定训练过程__：

    一些模型依赖梯度裁剪技术来裁剪训练中的梯度，以稳定训练过程。在 v2 中，梯度裁剪配置在 `optim_wrapper` 中指定：

    ```python
    optim_wrapper = dict(
        optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
        clip_grad=dict(max_norm=35, norm_type=2))
    ```

    如果您的配置继承了一个已经设置了 `optim_wrapper` 的基础配置，那么您可能需要 `_delete_=True` 字段来覆盖基础配置中无用的设置。详见配置文件的[说明文档](https://mmdetection.readthedocs.io/en/latest/tutorials/config.html)。

- __使用参数调度器 (param scheduler) 来加速模型收敛__：

    在 v2 中，学习率和动量调度统一使用 `param_scheduler` 配置。比如说，如下配置文件在 3D 检测中被用于加速模型收敛。
    更多细节详见 [MMEngine ParamScheduler](https://mmengine.readthedocs.io/en/latest/tutorials/param_scheduler.html) 的文档。

    ```python
    param_scheduler = [
        dict(
            type='CosineAnnealingLR',
            T_max=8,
            eta_min=0,
            by_epoch=True),
        dict(
            type='CosineAnnealingMomentum',
            T_max=8,
            eta_min=0.85 / 0.95,
            by_epoch=True),
    ]
    ```

## 自定义训练规程

默认情况，我们使用阶梯式学习率衰减的 1 倍训练规程。在 v2 中，学习率调度使用 `param_scheduler` 配置。
我们在 [MMEngine](https://mmengine.readthedocs.io/en/latest/tutorials/param_scheduler.html) 中支持很多学习率规划方案，比如`余弦退火`和`多项式衰减`规程。下面是一些样例：

- 多项式衰减规程:

    ```python
    param_scheduler = [
        dict(type='PolyLR', power=0.9, eta_min=1e-4, by_epoch=False)]
    ```

- 余弦退火规程:

    ```python
    param_scheduler = [
        dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
        dict(type='CosineAnnealingLR', eta_min_ratio=1e-5, by_epoch=True)]
    ```

## 自定义训练循环

在 v2 中，训练循环通过 `train_cfg`、`val_cfg` 和 `test_cfg` 配置。默认情况下使用基于 epoch 的训练循环：

```python
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict()
test_cfg = dict()
```

`val_interval` 控制验证的频率。如果不需要验证，可以省略 `val_cfg`。

## 自定义钩子

### 自定义并实现钩子

#### 1. 实现一个新钩子

存在一些情况下用户可能需要实现新钩子。在版本 v2.3.0 之后，MMDetection 支持自定义训练过程中的钩子 (#3395)。因此用户可以直接在 mmdet 中，或者在其基于 mmdet 的代码库中实现钩子并通过更改训练配置来使用钩子。
在 v2.3.0 之前，用户需要更改代码以使得训练开始之前钩子已经注册完毕。
这里我们给出一个，在 mmdet3d 中创建并使用新钩子的例子。

```python
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


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

取决于钩子的功能，用户需要指定钩子在每个训练阶段时的行为，具体包括如下阶段：`before_run`，`after_run`，`before_train_epoch`，`after_train_epoch`，`before_train_iter`，和 `after_train_iter`。

#### 2. 注册新钩子

接下来我们需要引入 `MyHook`。假设新钩子位于文件 `mmdet3d/core/utils/my_hook.py` 中，有两种方法可以实现之：

- 更改 `mmdet3d/core/utils/__init__.py` 来引入之：

    新定义的模块应在 `mmdet3d/core/utils/__init__.py` 中引入，以使得注册器可以找到新模块并注册之：

```python
from .my_hook import MyHook

__all__ = [..., 'MyHook']

```

或者在配置中使用 `custom_imports` 来人为地引入之

```python
custom_imports = dict(imports=['mmdet3d.core.utils.my_hook'], allow_failed_imports=False)
```

#### 3. 更改配置文件

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value)
]
```

您可以将字段 `priority` 设置为 `'NORMAL'` 或者 `'HIGHEST'`，来设置钩子的优先级，如下所示：

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value, priority='NORMAL')
]
```

默认情况，在注册阶段钩子的优先级被设置为 `NORMAL`。

### 使用 MMCV 中实现的钩子

如果钩子已经在 MMCV 中被实现了，您可以直接通过更改配置文件来使用该钩子：

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value, priority='NORMAL')
]
```

### 更改默认的运行时钩子

在 v2 中，默认钩子统一在 `default_hooks` 中配置：

```python
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)
```

#### 检查点配置

用户可以设置 `max_keep_ckpts` 来保存一定少量的检查点，或者用 `save_optimizer` 来决定是否保存优化器的状态。更多参数的细节详见 [MMEngine 文档](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.hooks.CheckpointHook.html)。

```python
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))
```

#### 日志配置

日志后端通过 `visualizer` 和 `vis_backends` 配置。现在 MMEngine 支持 `TensorboardVisBackend`，`WandbVisBackend` 等。

```python
vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(type='Det3DLocalVisualizer', vis_backends=vis_backends)
```

#### 评估配置

在 v2 中，评估器通过 `val_evaluator` 配置：

```python
val_evaluator = dict(type='NuScenesMetric')
```
