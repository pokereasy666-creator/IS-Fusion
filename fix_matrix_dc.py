filepath = "mmdet3d/models/middle_encoders/fusion_encoder.py"
with open(filepath, "r") as f:
    lines = f.readlines()

new_lines = []
skip_next = False

for i in range(len(lines)):
    line = lines[i]
    
    # 找到引发崩溃的 cur_img_aug_matrix 这一行
    if "cur_img_aug_matrix = img_aug_matrix[b] if not isinstance(img_aug_matrix, list) else img_aug_matrix[0][b]" in line:
        
        # 插入我们强大的解析函数
        safe_code = """
            def _get_item(matrix_var, batch_idx):
                val = matrix_var
                while isinstance(val, list) and len(val) == 1: val = val[0]
                if hasattr(val, 'data'): val = val.data
                while isinstance(val, list) and len(val) == 1: val = val[0]
                if hasattr(val, 'data'): val = val.data
                if isinstance(val, list) or isinstance(val, torch.Tensor):
                    return val[batch_idx]
                return val

            cur_img_aug_matrix = _get_item(img_aug_matrix, b)
            cur_lidar2img = _get_item(lidar2img, b)
"""
        new_lines.append(safe_code)
        
        # 跳过旧的这两行
        skip_next = True
        continue
        
    if skip_next and "cur_lidar2img = lidar2img[b]" in line:
        skip_next = False
        continue
        
    new_lines.append(line)

with open(filepath, "w") as f:
    f.writelines(new_lines)

print("✅ fix_matrix_dc.py 执行成功：顽固的 cur_img_aug_matrix 已被彻底抹除替换！")
