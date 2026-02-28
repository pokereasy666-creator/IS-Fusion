filepath = "mmdet3d/models/dense_heads/transfusion_head_v2.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

old_code = """        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes"""

new_code = """        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes
            # --- 核心修复：把真实标签同步到 GPU ---
            gt_labels_3d = gt_labels_3d.to(labels.device)"""

if "核心修复：把真实标签同步到 GPU" not in code:
    code = code.replace(old_code, new_code)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    print("✅ 标签设备同步完成！CPU -> GPU 对齐成功！")
else:
    print("⚠️ 已经同步过了。")
