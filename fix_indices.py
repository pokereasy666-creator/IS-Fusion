import re

filepath = "mmdet3d/core/bbox/assigners/hungarian_assigner.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# 确保把 cost 转为 CPU numpy，以防它在 GPU 上
if "cost.detach().cpu().numpy()" not in code:
    code = code.replace(
        'cost = np.nan_to_num(cost, nan=1e5, posinf=1e5, neginf=-1e5)',
        '''cost_np = cost.detach().cpu().numpy() if hasattr(cost, 'detach') else cost
        cost = np.nan_to_num(cost_np, nan=1e5, posinf=1e5, neginf=-1e5)'''
    )

# 确保把返回的 Numpy 数组转回 PyTorch Tensor
if "torch.as_tensor(matched_row_inds" not in code:
    code = re.sub(
        r'(matched_row_inds, matched_col_inds = linear_sum_assignment\(.*?\))',
        r'\1\n        matched_row_inds = torch.as_tensor(matched_row_inds, dtype=torch.long, device=assigned_labels.device)\n        matched_col_inds = torch.as_tensor(matched_col_inds, dtype=torch.long, device=assigned_labels.device)',
        code
    )

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)

print("✅ 成功修补索引类型！已将 NumPy 数组安全转换为 PyTorch Tensor。")
