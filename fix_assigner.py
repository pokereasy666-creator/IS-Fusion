filepath = "mmdet3d/core/bbox/assigners/hungarian_assigner.py"

with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# 定位到出问题的这一行
old_line = "cls_cost = self.cls_cost(cls_pred[0].T, gt_labels)"

# 替换成兼容 MMDetection 3.x 的代码
# 在 MMDetection 3.x 中，FocalLossCost 可以直接通过内部函数计算 Tensor 损失，
# 但为了绝对安全，我们直接手写 FocalLossCost 的计算逻辑来替换掉有兼容性问题的接口调用。
# 这是标准的 Focal Loss 匹配代价计算：
new_code = """
        # --- Compatibility Patch for MMDetection 3.x ---
        # Instead of calling self.cls_cost which expects InstanceData, 
        # we compute the focal loss cost directly on tensors.
        alpha = self.cls_cost.alpha
        gamma = self.cls_cost.gamma
        cls_pred_T = cls_pred[0].T
        neg_cost = -(1 - cls_pred_T + 1e-8).log() * (1 - alpha) * cls_pred_T**gamma
        pos_cost = -(cls_pred_T + 1e-8).log() * alpha * (1 - cls_pred_T)**gamma
        cls_cost = (pos_cost[:, gt_labels] - neg_cost[:, gt_labels]) * self.cls_cost.weight
        # -----------------------------------------------
"""

if old_line in code:
    code = code.replace(old_line, new_code)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    print("✅ 成功修补 hungarian_assigner.py！绕过了 MMDetection 3.x 的接口变更。")
else:
    print("⚠️ 未找到旧代码，请检查文件内容是否已被修改。")
