filepath = "configs/isfusion/isfusion_0075voxel_8gb.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# 将错误缩减的 64 改回原本对齐图像特征的 128
code = code.replace("embed_dims=64", "embed_dims=128")

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)

print("✅ fusion_encoder 内部维度已修正回 128，Tensor 变形矩阵已对齐！")
