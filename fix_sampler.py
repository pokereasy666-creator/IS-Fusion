import re

filepath = "mmdet3d/models/dense_heads/transfusion_head_v2.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# 找到调用 sampler 的地方
old_code_block = """        sampling_result = self.bbox_sampler.sample(
            assign_result,
            bboxes,
            gt_bboxes_3d,
            gt_labels_3d,
            type='3D')"""

# 直接手动构造 SamplingResult，这是 PseudoSampler 在背后做的唯一事情
new_code_block = """        # --- MMDetection 3.x Compatibility Patch for Sampler ---
        # Instead of calling self.bbox_sampler.sample which expects InstanceData,
        # we manually create a SamplingResult (PseudoSampler does exactly this).
        from mmdet.models.task_modules.samplers.sampling_result import SamplingResult
        
        # MMDetection 3.x SamplingResult init signature:
        # __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags)
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        gt_flags = bboxes.new_zeros((bboxes.shape[0],), dtype=torch.uint8)
        
        sampling_result = SamplingResult(
            pos_inds=pos_inds,
            neg_inds=neg_inds,
            bboxes=bboxes,
            gt_bboxes=gt_bboxes_3d,
            assign_result=assign_result,
            gt_flags=gt_flags
        )
        # -------------------------------------------------------"""

if old_code_block in code:
    code = code.replace(old_code_block, new_code_block)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    print("✅ 成功修补 transfusion_head_v2.py！已绕过 MMDetection 3.x 的 Sampler 接口变更。")
else:
    print("⚠️ 找不到精确匹配的 Sampler 代码块，尝试备用替换方案...")
    # 备用正则匹配
    code = re.sub(
        r'sampling_result\s*=\s*self\.bbox_sampler\.sample\(\s*assign_result,\s*bboxes,\s*gt_bboxes_3d,\s*gt_labels_3d,\s*type=\'3D\'\s*\)',
        new_code_block,
        code,
        flags=re.MULTILINE
    )
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    print("✅ 备用方案修补完成。")
