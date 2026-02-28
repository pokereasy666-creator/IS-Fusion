import os

filepath = "configs/isfusion/isfusion_0075voxel_8gb.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# 暴力替换：将训练和验证时的体素峰值全部下调 1/3
code = code.replace("120000", "80000")
code = code.replace("60000", "40000")

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)

print("✅ 体素防波堤【暴力版】已部署！120000 -> 80000, 60000 -> 40000。这次绝对生效！")
