# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import mmdet
import mmseg
from .version import __version__, short_version


def digit_version(version_str):
    digit_version = []
    for x in version_str.split('.'):
        if x.isdigit():
            digit_version.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            digit_version.append(int(patch_version[0]) - 1)
            digit_version.append(int(patch_version[1]))
    return digit_version


# OpenMMLab v2 version requirements
mmcv_minimum_version = '2.0.0'
mmcv_maximum_version = '2.2.0'
mmcv_version = digit_version(mmcv.__version__)
assert (mmcv_version >= digit_version(mmcv_minimum_version)
        and mmcv_version <= digit_version(mmcv_maximum_version)), \
    f'MMCV=={mmcv.__version__} is used but incompatible. ' \
    f'Please install mmcv>={mmcv_minimum_version}, <={mmcv_maximum_version}.'

mmdet_minimum_version = '3.0.0'
mmdet_maximum_version = '3.4.0'
mmdet_version = digit_version(mmdet.__version__)
assert (mmdet_version >= digit_version(mmdet_minimum_version)
        and mmdet_version <= digit_version(mmdet_maximum_version)), \
    f'MMDET=={mmdet.__version__} is used but incompatible. ' \
    f'Please install mmdet>={mmdet_minimum_version}, <={mmdet_maximum_version}.'

mmseg_minimum_version = '1.0.0'
mmseg_maximum_version = '1.3.0'
mmseg_version = digit_version(mmseg.__version__)
assert (mmseg_version >= digit_version(mmseg_minimum_version)
        and mmseg_version <= digit_version(mmseg_maximum_version)), \
    f'MMSEG=={mmseg.__version__} is used but incompatible. ' \
    f'Please install mmseg>={mmseg_minimum_version}, <={mmseg_maximum_version}.'

# Apply v2 compatibility shims (must come before other mmdet3d imports)
from mmdet3d import compat  # noqa: F401

# Initialize registries
from mmdet3d import registry  # noqa: F401


def register_all_modules(init_default_scope=True):
    """Register all modules in mmdet3d into the registries.

    Args:
        init_default_scope (bool): Whether to initialize the default scope.
            Defaults to True. When used with other OpenMMLab projects,
            set to False to avoid conflicts.
    """
    import mmdet3d.models  # noqa: F401
    import mmdet3d.datasets  # noqa: F401
    import mmdet3d.core  # noqa: F401
    import mmdet3d.evaluation  # noqa: F401
    import mmdet3d.engine  # noqa: F401
    import mmdet3d.visualization  # noqa: F401

    if init_default_scope:
        from mmengine.registry import DefaultScope
        never_created = DefaultScope.get_current_instance() is None \
            or not DefaultScope.check_instance_created('mmdet3d')
        if never_created:
            DefaultScope.get_instance('mmdet3d', scope_name='mmdet3d')


__all__ = ['__version__', 'short_version', 'register_all_modules']
