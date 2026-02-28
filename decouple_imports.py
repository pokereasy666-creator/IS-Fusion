import re

filepath = "mmdet3d/models/dense_heads/transfusion_head_v2.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# 找到那段把大家绑在一起的旧代码
bad_block = """try:
    from mmdet3d.core import (xywhr2xyxyr, limit_period, PseudoSampler)
except ImportError:
    try:
        from mmdet3d.structures.bbox_3d import limit_period, xywhr2xyxyr
    except ImportError:
        from mmdet3d.models.utils import limit_period, xywhr2xyxyr
    try:
        from mmdet.models.task_modules.samplers import PseudoSampler
    except ImportError:
        from mmdet.core.bbox.samplers import PseudoSampler

try:
    from mmdet3d.core.bbox.structures import rotation_3d_in_axis
    from mmdet3d.core import Box3DMode, LiDARInstance3DBoxes
except ImportError:
    from mmdet3d.structures.bbox_3d import rotation_3d_in_axis
    from mmdet3d.structures import Box3DMode, LiDARInstance3DBoxes"""

# 拆分成独立的安全导入（谁也不连累谁）
good_block = """try:
    from mmdet3d.core import xywhr2xyxyr, limit_period
except ImportError:
    try:
        from mmdet3d.structures.bbox_3d import limit_period, xywhr2xyxyr
    except ImportError:
        from mmdet3d.models.utils import limit_period, xywhr2xyxyr

try:
    from mmdet.models.task_modules.samplers import PseudoSampler
except ImportError:
    try:
        from mmdet.core.bbox.samplers import PseudoSampler
    except ImportError:
        try:
            from mmdet.core import PseudoSampler
        except ImportError:
            class PseudoSampler: pass

try:
    from mmdet3d.core.bbox.structures import rotation_3d_in_axis
except ImportError:
    from mmdet3d.structures.bbox_3d import rotation_3d_in_axis

try:
    from mmdet3d.core import Box3DMode, LiDARInstance3DBoxes
except ImportError:
    from mmdet3d.structures import Box3DMode, LiDARInstance3DBoxes"""

if bad_block in code:
    code = code.replace(bad_block, good_block)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    print("✅ 成功解除导入连坐！大家现在各自独立加载了。")
else:
    # 备用正则匹配，防止缩进不同
    code = re.sub(r'try:\s*from mmdet3d\.core import \(xywhr2xyxyr.*?LiDARInstance3DBoxes', good_block, code, flags=re.DOTALL)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    print("✅ 成功解除导入连坐 (通过正则)！")
