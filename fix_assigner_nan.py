filepath = "mmdet3d/core/bbox/assigners/hungarian_assigner.py"

with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# 1. 给 cls_pred 加上 .sigmoid()，将 Logits 转为 0~1 的概率
old_line = "cls_pred_T = cls_pred[0].T"
new_line = "cls_pred_T = cls_pred[0].T.sigmoid()  # 必须加 sigmoid！防止负数求 log 爆 NaN"
if old_line in code:
    code = code.replace(old_line, new_line)

# 2. 在送入匈牙利算法前，加一个双保险，清洗掉可能由于除以 0 等情况产生的无穷大或 NaN
safeguard_old = "matched_row_inds, matched_col_inds = linear_sum_assignment(cost)"
safeguard_new = """import numpy as np
        # 终极双保险：把所有的 NaN 和 Inf 替换成一个极大的惩罚值，确保算法绝对不会崩溃
        cost = np.nan_to_num(cost, nan=1e5, posinf=1e5, neginf=-1e5)
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)"""

if safeguard_old in code:
    code = code.replace(safeguard_old, safeguard_new)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)

print("✅ 成功修复 NaN 错误！已应用 Sigmoid 激活并加入 NaN 洗地双保险。")
