filepath = "mmdet3d/models/dense_heads/transfusion_head_v2.py"

with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# 这是你文件里原本原封不动的 3 行代码
old_code = """        sampling_result = self.bbox_sampler.sample(
            assign_result_ensemble, bboxes_tensor, gt_bboxes_tensor
        )"""

# 我们用一个极简的自定义类完美平替它，避开一切 MMDetection 版本冲突
new_code = """        # --- MMDetection 3.x Compatibility Patch ---
        class CustomSamplingResult:
            def __init__(self, assign_result, bboxes, gt_bboxes):
                self.pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
                self.neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
                self.pos_assigned_gt_inds = assign_result.gt_inds[self.pos_inds] - 1
                self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds]

        sampling_result = CustomSamplingResult(assign_result_ensemble, bboxes_tensor, gt_bboxes_tensor)
        # -------------------------------------------"""

if old_code in code:
    code = code.replace(old_code, new_code)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    print("✅ Sampler 完美修补成功！已经用极简自定义类绕过了新旧 API 差异。")
else:
    print("⚠️ 找不到目标代码，可能是之前已经替换过了或者缩进不匹配。打开 transfusion_head_v2.py 第1121行确认一下。")

