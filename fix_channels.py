import re

filepath = "configs/isfusion/isfusion_0075voxel_8gb.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# Fix 1: pts_backbone in_channels should be 64 (it was 128)
code = re.sub(r'in_channels=128,\s*out_channels=\[64,\s*128\]', 
              'in_channels=64, out_channels=[64, 128]', code)

# Fix 2: fusion_encoder embed_dims should match (it outputs 64 to backbone now)
code = re.sub(r'fusion_encoder=dict\(\s*embed_dims=128,', 
              'fusion_encoder=dict(embed_dims=64,', code)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)

print("✅ Backbone 通道对齐完成！64 -> 64")
