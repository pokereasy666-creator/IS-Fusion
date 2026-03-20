# Copyright (c) OpenMMLab. All rights reserved.
import mmdet
import mmdet3d
import mmseg

try:
    from mmengine.utils import collect_env as collect_base_env
except ImportError:
    from mmcv.utils import collect_env as collect_base_env

try:
    from mmengine.utils import get_git_hash
except ImportError:
    def get_git_hash():
        return 'unknown'


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['MMDetection'] = mmdet.__version__
    env_info['MMSegmentation'] = mmseg.__version__

    try:
        git_hash = get_git_hash()
    except Exception:
        git_hash = 'unknown'

    env_info['MMDetection3D'] = mmdet3d.__version__ + '+' + git_hash[:7]
    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')
