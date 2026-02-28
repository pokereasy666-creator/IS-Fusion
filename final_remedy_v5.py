import re

filepath = "mmdet3d/models/dense_heads/transfusion_head_v2.py"
with open(filepath, "r", encoding="utf-8") as f:
    orig_code = f.read()

# 1. 彻底清除头部的 Import 乱象 (从第一行到 from torch import nn 之前全部重写)
new_header = """import copy
import numpy as np
import torch
from mmcv.cnn import ConvModule, build_conv_layer

# --- Robust Imports for MMCV 2.x / MMEngine ---
try:
    from mmengine.model.weight_init import kaiming_init
except ImportError:
    from mmcv.cnn import kaiming_init

def force_fp32(apply_to=None, out_fp16=False):
    def decorator(func): return func
    return decorator

# --- Inline Drawing Utils (Fixing ImportError forever) ---
def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    a1, b1, c1 = 1, (height + width), width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2
    a2, b2, c2 = 4, 2 * (height + width), (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2
    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (height + width), (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

def draw_heatmap_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = heatmap.new_tensor(np.ogrid[-radius:radius + 1, -radius:radius + 1])
    x, y = gaussian
    h = torch.exp(-(x ** 2 + y ** 2) / (2 * (diameter / 6) ** 2))
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    left, top = min(center[0], radius), min(center[1], radius)
    right, bottom = min(heatmap.shape[1] - center[0], radius + 1), min(heatmap.shape[0] - center[1], radius + 1)
    masked_heatmap  = heatmap[center[1] - top:center[1] + bottom, center[0] - left:center[0] + right]
    masked_gaussian = h[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

try:
    from mmdet3d.models.layers import circle_nms
except ImportError:
    from mmdet3d.ops.iou3d.iou3d_utils import circle_nms
# --------------------------------------------
"""

# 保留文件中从 "from torch import nn" 开始的所有后续内容
main_parts = orig_code.split("from torch import nn")
if len(main_parts) < 2:
    print("❌ 错误：无法识别文件结构，请确保文件没有被彻底删减！")
    exit(1)

final_code = new_header + "\nfrom torch import nn" + main_parts[1]

# 2. 修复 Sampler 逻辑 (替换掉导致 AttributeError 的那几行)
sampler_old = r'sampling_result = self\.bbox_sampler\.sample\([\s\S]*?gt_bboxes_tensor\s*\)'
sampler_new = """        # --- MMDetection 3.x Compatibility Bypass ---
        class CustomSR: pass
        sampling_result = CustomSR()
        sampling_result.pos_inds = torch.nonzero(assign_result_ensemble.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        sampling_result.neg_inds = torch.nonzero(assign_result_ensemble.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        sampling_result.pos_assigned_gt_inds = (assign_result_ensemble.gt_inds[sampling_result.pos_inds] - 1).long()
        sampling_result.pos_gt_bboxes = gt_bboxes_tensor[sampling_result.pos_assigned_gt_inds]"""

final_code = re.sub(sampler_old, sampler_new, final_code)

# 3. 修复推理阶段的 Box 类型
final_code = final_code.replace(
    'metas[0]["box_type_3d"](rets[0][0]["bboxes"], box_dim=rets[0][0]["bboxes"].shape[-1])',
    'rets[0][0]["bboxes"]'
)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(final_code)

print("✅ 深度微创手术成功完成！已适配新版导入并绕过 Sampler 冲突。")
