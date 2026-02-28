import re

filepath = "mmdet3d/models/middle_encoders/fusion_encoder.py"
with open(filepath, "r") as f:
    code = f.read()

# 定位到矩阵乘法发生的地方，并在前面强行加入 .to()
old_matmul = "cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)"
new_matmul = """
            if isinstance(cur_lidar2image, torch.Tensor):
                cur_lidar2image = cur_lidar2image.to(cur_coords.device).float()
            else:
                cur_lidar2image = torch.tensor(cur_lidar2image, device=cur_coords.device, dtype=torch.float32)
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
"""

if old_matmul in code:
    code = code.replace(old_matmul, new_matmul.strip())
    print("✅ 找到并修复了 matmul 运算前的设备不匹配问题！")
else:
    print("⚠️ 没找到旧的 matmul 代码，请检查。")

# 为了防患于未然，把后面可能发生的另一个平移加法也加上设备同步
old_add = "cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 1)"
new_add = """
            cur_coords += cur_lidar2image[:, :3, 3].to(cur_coords.device).reshape(-1, 1)
"""

if old_add in code:
    code = code.replace(old_add, new_add.strip())
    print("✅ 找到并修复了加法运算前的设备不匹配问题！")

with open(filepath, "w") as f:
    f.write(code)

