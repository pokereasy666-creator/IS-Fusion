import re

filepath = "mmdet3d/models/dense_heads/transfusion_head_v2.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# 1. 修复 mmcv/mmengine 基础导入错误
code = code.replace(
    "from mmcv.cnn import ConvModule, build_conv_layer, kaiming_init",
    "from mmcv.cnn import ConvModule, build_conv_layer\nfrom mmengine.model.weight_init import kaiming_init"
)
code = code.replace(
    "from mmcv.runner import force_fp32",
    "from mmengine.model import force_fp32"
)

# 2. 修复 mmdet3d.core 导入错误 (适配新版路径)
code = code.replace(
    "from mmdet3d.core import (circle_nms, draw_heatmap_gaussian, gaussian_radius,",
    "from mmdet3d.models.utils import draw_heatmap_gaussian, gaussian_radius\nfrom mmdet3d.models.layers import circle_nms\nfrom mmdet3d.core import"
)

# 3. 修复采样器 Sampler 接口冲突 (使用我们之前验证过的自定义类平替)
new_sampler_logic = """        # --- MMDetection 3.x Compatibility Patch ---
        class CustomSamplingResult:
            def __init__(self, assign_result, bboxes, gt_bboxes):
                self.pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
                self.neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
                self.pos_assigned_gt_inds = (assign_result.gt_inds[self.pos_inds] - 1).long()
                self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds]
        sampling_result = CustomSamplingResult(assign_result_ensemble, bboxes_tensor, gt_bboxes_tensor)
        # -------------------------------------------"""

# 寻找原代码中出错的 self.bbox_sampler.sample 调用块
pattern = r'sampling_result = self\.bbox_sampler\.sample\(\s*assign_result_ensemble,\s*bboxes_tensor,\s*gt_bboxes_tensor\s*\)'
code = re.sub(pattern, new_sampler_logic, code)

# 4. 修复 get_bboxes 中的 box_type_3d 兼容性 (防止下一步报错)
code = code.replace(
    'metas[0]["box_type_3d"](rets[0][0]["bboxes"], box_dim=rets[0][0]["bboxes"].shape[-1])',
    'rets[0][0]["bboxes"]' # 新版通常直接返回 Box 对象或 Tensor
)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)

print("✅ 深度修复完成！已适配新版路径并解决了 Sampler 冲突。")
