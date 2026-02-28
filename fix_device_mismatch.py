import re

filepath = "mmdet3d/models/dense_heads/transfusion_head_v2.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# 匹配我们之前的自定义 Sampler 补丁
old_block = r'# --- 彻底绕过 Sampler 的 InstanceData 报错 ---.*?sampling_result\.pos_gt_bboxes = gt_bboxes_tensor\[sampling_result\.pos_assigned_gt_inds\]'

new_block = """# --- 彻底绕过 Sampler 的 InstanceData 报错 (包含 Device 同步) ---
        target_device = assign_result_ensemble.gt_inds.device
        gt_bboxes_tensor = gt_bboxes_tensor.to(target_device)
        gt_labels_3d = gt_labels_3d.to(target_device)
        
        class CustomSR: pass
        sampling_result = CustomSR()
        sampling_result.pos_inds = torch.nonzero(assign_result_ensemble.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        sampling_result.neg_inds = torch.nonzero(assign_result_ensemble.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        sampling_result.pos_assigned_gt_inds = (assign_result_ensemble.gt_inds[sampling_result.pos_inds] - 1).long()
        sampling_result.pos_gt_bboxes = gt_bboxes_tensor[sampling_result.pos_assigned_gt_inds]"""

if "包含 Device 同步" not in code:
    code = re.sub(old_block, new_block, code, flags=re.DOTALL)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    print("✅ 设备同步补丁已注入！CPU / GPU 张量已完全对齐！")
else:
    print("⚠️ 已经注入过了。")
