import re

filepath = "mmdet3d/models/dense_heads/transfusion_head_v2.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# 用自定义类极简替换，避开 MMDetection 3.x InstanceData 接口冲突
new_code = """# --- MMDetection 3.x Compatibility Patch ---
        class CustomSamplingResult:
            def __init__(self, assign_result, bboxes, gt_bboxes):
                self.pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
                self.neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
                self.pos_assigned_gt_inds = assign_result.gt_inds[self.pos_inds] - 1
                self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds]

        sampling_result = CustomSamplingResult(assign_result_ensemble, bboxes_tensor, gt_bboxes_tensor)
        # -------------------------------------------"""

# 精准正则匹配：只替换 sampling_result = self.bbox_sampler.sample(...) 这一句话
code, num_subs = re.subn(
    r'sampling_result\s*=\s*self\.bbox_sampler\.sample\([^)]+\)',
    new_code.strip(),
    code
)

if num_subs > 0:
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"✅ Sampler 完美修补成功！(替换了 {num_subs} 处)")
else:
    print("⚠️ 找不到目标代码，请确保文件已经恢复到了原始状态！")

