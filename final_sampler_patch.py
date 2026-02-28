filepath = "mmdet3d/models/dense_heads/transfusion_head_v2.py"
with open(filepath, "r") as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if "sampling_result = self.bbox_sampler.sample(" in line:
        skip = True
        new_lines.append("        # --- MMDetection 3.x Compatibility Patch ---\n")
        new_lines.append("        class CustomSR: pass\n")
        new_lines.append("        sampling_result = CustomSR()\n")
        new_lines.append("        sampling_result.pos_inds = torch.nonzero(assign_result_ensemble.gt_inds > 0, as_tuple=False).squeeze(-1).unique()\n")
        new_lines.append("        sampling_result.neg_inds = torch.nonzero(assign_result_ensemble.gt_inds == 0, as_tuple=False).squeeze(-1).unique()\n")
        new_lines.append("        sampling_result.pos_assigned_gt_inds = (assign_result_ensemble.gt_inds[sampling_result.pos_inds] - 1).long()\n")
        new_lines.append("        sampling_result.pos_gt_bboxes = gt_bboxes_tensor[sampling_result.pos_assigned_gt_inds]\n")
        continue
    if skip and ")" in line:
        skip = False
        continue
    if not skip:
        new_lines.append(line)

with open(filepath, "w") as f:
    f.writelines(new_lines)
print("✅ Sampler 逻辑已安全重写。")
