filepath = "mmdet3d/models/middle_encoders/fusion_encoder.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# 原本出问题的代码行
old_line = "cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul("

# 我们把这个微小矩阵移到 CPU 上求逆，然后再放回 GPU 做乘法
new_line = "cur_coords = cur_lidar_aug_matrix[:3, :3].cpu().inverse().to(cur_lidar_aug_matrix.device).matmul("

if old_line in code:
    code = code.replace(old_line, new_line)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    print("✅ 成功修补了 torch.inverse，已绕过 cuSOLVER 的显存瓶颈！")
else:
    print("⚠️ 没找到需要替换的代码行，请手动检查 fusion_encoder.py。")
