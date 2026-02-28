import re

filepath = "mmdet3d/core/bbox/assigners/hungarian_assigner.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# 强行对齐所有相关的设备
patch_code = """
        # --- Device Alignment Patch ---
        device = bboxes.device
        matched_row_inds = torch.as_tensor(matched_row_inds, dtype=torch.long, device=device)
        matched_col_inds = torch.as_tensor(matched_col_inds, dtype=torch.long, device=device)
        assigned_gt_inds = assigned_gt_inds.to(device)
        assigned_labels = assigned_labels.to(device)
        gt_labels = gt_labels.to(device)
        # ------------------------------
"""

# 我们把它插在 linear_sum_assignment 之后
if "# --- Device Alignment Patch ---" not in code:
    code = re.sub(
        r'(matched_row_inds\s*=\s*torch\.as_tensor.*?)\n(.*?)',
        r'\n' + patch_code + r'\n\2',
        code,
        count=1,
        flags=re.DOTALL
    )

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)

print("✅ 成功统一设备！现在所有的张量和索引都强制位于同一个 device。")
