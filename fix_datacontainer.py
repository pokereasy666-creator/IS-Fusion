filepath = "mmdet3d/models/detectors/isfusion.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# 我们需要找到 forward_pts_train 方法里准备 loss_inputs 的地方
old_code = """
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
"""

# 在放入 loss_inputs 之前，检查并解包 DataContainer
new_code = """
        # 解包 DataContainer
        if hasattr(gt_bboxes_3d[0], 'data'):
            gt_bboxes_3d = [b.data for b in gt_bboxes_3d]
        if hasattr(gt_labels_3d[0], 'data'):
            gt_labels_3d = [l.data for l in gt_labels_3d]
            
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
"""

if old_code in code:
    code = code.replace(old_code, new_code)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    print("✅ 成功修补 isfusion.py！DataContainer 已解包。")
else:
    print("⚠️ 没找到精确的替换位置，尝试备用替换方案...")
    # 备用替换方案，更宽泛的匹配
    import re
    code = re.sub(r'loss_inputs\s*=\s*\[gt_bboxes_3d,\s*gt_labels_3d,\s*outs\]', 
                  r'''
        if hasattr(gt_bboxes_3d[0], 'data'):
            gt_bboxes_3d = [b.data for b in gt_bboxes_3d]
        if hasattr(gt_labels_3d[0], 'data'):
            gt_labels_3d = [l.data for l in gt_labels_3d]
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]''', code)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    print("✅ 备用方案修补完成。")
