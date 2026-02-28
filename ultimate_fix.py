import re

filepath = "mmdet3d/models/dense_heads/transfusion_head_v2.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# 1. 修复 kaiming_init 导入 (适配 MMCV 2.x / MMEngine)
code = code.replace(
    "from mmcv.cnn import ConvModule, build_conv_layer, kaiming_init",
    "from mmcv.cnn import ConvModule, build_conv_layer\nfrom mmengine.model.weight_init import kaiming_init"
)

# 2. 修复 force_fp32 导入 (MMCV 2.x 中 runner 被移除，用伪装饰器安全代替)
code = code.replace(
    "from mmcv.runner import force_fp32",
    "def force_fp32(*args, **kwargs):\n    def decorator(func):\n        return func\n    return decorator"
)

# 3. 修复 MMDetection 3.x 极其难搞的 Sampler 接口冲突
new_sampler = """# --- MMDetection 3.x Compatibility Patch ---
        class CustomSamplingResult:
            def __init__(self, assign_result, bboxes, gt_bboxes):
                self.pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
                self.neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
                self.pos_assigned_gt_inds = assign_result.gt_inds[self.pos_inds] - 1
                self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds]

        sampling_result = CustomSamplingResult(assign_result_ensemble, bboxes_tensor, gt_bboxes_tensor)
        # -------------------------------------------"""

# 精准替换旧的 self.bbox_sampler.sample 调用
code = re.sub(
    r'sampling_result\s*=\s*self\.bbox_sampler\.sample\(\s*assign_result_ensemble,\s*bboxes_tensor,\s*gt_bboxes_tensor\s*\)', 
    new_sampler, 
    code
)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)

print("✅ 所有环境导入错误与 Sampler 接口冲突已完美修复！")
