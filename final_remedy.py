import re
import os

path = "mmdet3d/models/dense_heads/transfusion_head_v2.py"

with open(path, 'r', encoding='utf-8') as f:
    code = f.read()

# 1. 修复顶部导入 (适配 MMCV 2.x / mmengine)
code = code.replace(
    "from mmcv.cnn import ConvModule, build_conv_layer, kaiming_init",
    "from mmcv.cnn import ConvModule, build_conv_layer\nfrom mmengine.model.weight_init import kaiming_init"
)
code = code.replace("from mmcv.runner import force_fp32", "from mmengine.model import force_fp32")

# 2. 注入丢失的绘图工具函数 (既然库里找不到，我们就直接写进代码里)
drawing_utils = """
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
"""
# 在第一个 import 后面注入
code = code.replace("import torch", "import torch\n" + drawing_utils)

# 3. 修复 circle_nms 的导入位置
code = code.replace(
    "from mmdet3d.core import (circle_nms, draw_heatmap_gaussian, gaussian_radius,",
    "from mmdet3d.models.layers import circle_nms\nfrom mmdet3d.core import ("
)

# 4. 彻底重写 Sampler 逻辑 (绕过 InstanceData 报错)
sampler_pattern = r'sampling_result = self\.bbox_sampler\.sample\([\s\S]*?gt_bboxes_tensor\s*\)'
new_sampler_logic = """        # --- MMDetection 3.x Compatibility Bypass ---
        from mmdet.models.task_modules.samplers.sampling_result import SamplingResult
        pos_inds = torch.nonzero(assign_result_ensemble.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assign_result_ensemble.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes_tensor, gt_bboxes_tensor, assign_result_ensemble, bboxes_tensor.new_zeros(bboxes_tensor.shape[0], dtype=torch.uint8))
        sampling_result.pos_assigned_gt_inds = (assign_result_ensemble.gt_inds[pos_inds] - 1).long()
        sampling_result.pos_gt_bboxes = gt_bboxes_tensor[sampling_result.pos_assigned_gt_inds]"""

code = re.sub(sampler_pattern, new_sampler_logic, code)

# 5. 修复最后的 Box 类型调用错误
code = code.replace(
    'metas[0]["box_type_3d"](rets[0][0]["bboxes"], box_dim=rets[0][0]["bboxes"].shape[-1])',
    'rets[0][0]["bboxes"]'
)

with open(path, 'w', encoding='utf-8') as f:
    f.write(code)

print("✅ 恭喜！深度微创手术已完成。")
print("✅ 导入路径已修正，绘图函数已内联，采样器已适配。")
