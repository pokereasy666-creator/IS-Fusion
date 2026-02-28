import os
import re

filepath = "configs/isfusion/isfusion_0075voxel_8gb.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# 将训练时的最大体素从 60000 压低到 45000，限制极端场景的显存突刺
code = re.sub(r'max_voxels\s*=\s*\(\s*60000\s*,\s*120000\s*,?\s*\)', 'max_voxels=(45000, 90000)', code)
code = re.sub(r'max_voxels\s*=\s*\[\s*60000\s*,\s*120000\s*,?\s*\]', 'max_voxels=[45000, 90000]', code)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)

print("✅ 体素防波堤已筑起！已将最大体素数量下调至 45000，彻底锁死显存上限！")
