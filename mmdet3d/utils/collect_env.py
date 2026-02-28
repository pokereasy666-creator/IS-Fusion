# Copyright (c) OpenMMLab. All rights reserved.
import mmdet
import mmdet3d
import mmseg

# ----------------- 物理修复开始：适配 MMCV / MMEngine -----------------
try:
    # 尝试旧版路径 (MMCV 1.x)
    from mmcv.utils import collect_env as collect_base_env
    from mmcv.utils import get_git_hash
except ImportError:
    # 新版路径 (MMEngine / MMCV 2.x)
    from mmengine.utils import get_git_hash
    from mmengine.utils.dl_utils import collect_env as collect_base_env
# ----------------- 物理修复结束 -----------------


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['MMDetection'] = mmdet.__version__
    env_info['MMSegmentation'] = mmseg.__version__
    
    # 增加一点鲁棒性，防止获取 git hash 失败报错
    try:
        git_hash = get_git_hash()
    except Exception:
        git_hash = 'unknown'

    env_info['MMDetection3D'] = mmdet3d.__version__ + '+' + git_hash[:7]

    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')
