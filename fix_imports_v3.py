filepath = "mmdet3d/models/dense_heads/transfusion_head_v2.py"
with open(filepath, "r") as f:
    lines = f.readlines()

new_lines = []
skip_mode = False

# 定义一个鲁棒性极强的导入块
robust_imports = """
# --- Robust Version Compatibility Imports ---
try:
    from mmcv.cnn import ConvModule, build_conv_layer, kaiming_init
except ImportError:
    from mmcv.cnn import ConvModule, build_conv_layer
    from mmengine.model.weight_init import kaiming_init

try:
    from mmcv.runner import force_fp32
except ImportError:
    try:
        from mmengine.model import force_fp32
    except ImportError:
        def force_fp32(*args, **kwargs):
            def decorator(f): return f
            return decorator

try:
    from mmdet3d.core import draw_heatmap_gaussian, gaussian_radius, circle_nms
except ImportError:
    try:
        from mmdet3d.models.utils import draw_heatmap_gaussian, gaussian_radius
        from mmdet3d.models.layers import circle_nms
    except ImportError:
        # 针对最新版 MMDetection3D 1.1+ 的路径
        from mmdet3d.utils import draw_heatmap_gaussian, gaussian_radius
        from mmdet3d.models.layers import circle_nms
# --------------------------------------------
"""

for line in lines:
    # 跳过原本那些冲突的导入行 (第4行到第20行左右)
    if "from mmcv.cnn import" in line or "from mmcv.runner import" in line or "from mmdet3d.core import (" in line:
        if not skip_mode:
            new_lines.append(robust_imports)
            skip_mode = True
        continue
    
    # 停止跳过，保留后续其他导入
    if skip_mode and ("from mmdet3d.core.bbox.structures" in line or "from mmdet3d.core import Box3DMode" in line):
        skip_mode = False
    
    if not skip_mode:
        new_lines.append(line)

with open(filepath, "w") as f:
    f.writelines(new_lines)
print("✅ 导入路径已重写为自适应模式，将自动匹配新老版本。")
