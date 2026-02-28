import os

filepath = "mmdet3d/models/dense_heads/transfusion_head_v2.py"
with open(filepath, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if "sampling_result = self.bbox_sampler.sample(" in line:
        skip = True
        new_lines.append("""        # --- MMDetection 3.x Compatibility Patch for Sampler ---
        from mmdet.models.task_modules.samplers.sampling_result import SamplingResult
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        gt_flags = bboxes.new_zeros((bboxes.shape[0],), dtype=torch.uint8)
        sampling_result = SamplingResult(
            pos_inds=pos_inds, neg_inds=neg_inds, bboxes=bboxes, gt_bboxes=gt_bboxes_3d, 
            assign_result=assign_result, gt_flags=gt_flags)
        # -------------------------------------------------------\n""")
        continue
    
    if skip:
        # 忽略掉原本的 sampler.sample 多行传参
        if "type='3D')" in line or "type=\"3D\")" in line:
            skip = False
        continue

    new_lines.append(line)

with open(filepath, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("✅ 暴力修补 Sampler 完成！已经彻底挖掉了出错的旧接口。")
