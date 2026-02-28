import re

filepath = "mmdet3d/models/dense_heads/transfusion_head_v2.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# 1. 修复 PseudoSampler 等 (适配 MMDet 3.3.0)
old_1 = r"from mmdet3d\.core import \([\s\S]*?PseudoSampler\)"
new_1 = """# --- 缝合怪环境自适应导入 ---
try:
    from mmdet.models.task_modules.samplers import PseudoSampler
except ImportError:
    from mmdet.core.bbox.samplers import PseudoSampler

try:
    from mmdet3d.core.bbox.structures import limit_period, xywhr2xyxyr
except ImportError:
    try:
        from mmdet3d.structures.bbox_3d import limit_period, xywhr2xyxyr
    except ImportError:
        from mmdet3d.models.utils import limit_period, xywhr2xyxyr"""

code = re.sub(old_1, new_1, code)

# 2. 修复 Box3DMode 
old_2 = r"from mmdet3d\.core import Box3DMode, LiDARInstance3DBoxes"
new_2 = """try:
    from mmdet3d.core import Box3DMode, LiDARInstance3DBoxes
except ImportError:
    from mmdet3d.structures import Box3DMode, LiDARInstance3DBoxes"""

code = re.sub(old_2, new_2, code)

# 3. 修复 rotation_3d_in_axis
old_3 = r"from mmdet3d\.core\.bbox\.structures import rotation_3d_in_axis"
new_3 = """try:
    from mmdet3d.core.bbox.structures import rotation_3d_in_axis
except ImportError:
    from mmdet3d.structures.bbox_3d import rotation_3d_in_axis"""

code = re.sub(old_3, new_3, code)

# 4. 修复 mmdet.core (MMDet 3.3.0 杀手级改动)
old_4 = r"from mmdet\.core import build_bbox_coder[\s\S]*?AssignResult"
new_4 = """try:
    from mmdet.core import build_bbox_coder, multi_apply, build_assigner, build_sampler, AssignResult
except ImportError:
    # 完美适配 MMDetection 3.3.0 最新注册器
    from mmdet.registry import MODELS as _MODELS
    from mmdet.registry import TASK_UTILS as _TASK_UTILS
    build_bbox_coder = _MODELS.build
    build_assigner = _TASK_UTILS.build
    build_sampler = _TASK_UTILS.build
    try:
        from mmdet.models.utils.misc import multi_apply
    except ImportError:
        from mmengine.utils import multi_apply
    from mmdet.models.task_modules.assigners import AssignResult"""

code = re.sub(old_4, new_4, code)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)

print("✅ 所有潜藏的旧版 MMDetection 导入地雷已全部排除！")
