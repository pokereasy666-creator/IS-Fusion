filepath = "mmdet3d/core/bbox/assigners/hungarian_assigner.py"

with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# 把极其严格的 from_numpy 替换为自适应的 as_tensor
code = code.replace("torch.from_numpy(matched_row_inds)", "torch.as_tensor(matched_row_inds)")
code = code.replace("torch.from_numpy(matched_col_inds)", "torch.as_tensor(matched_col_inds)")

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)

print("✅ 成功修复类型转换冲突！将严格的 from_numpy 替换为了自适应的 as_tensor。")
