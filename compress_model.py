import re

filepath = "configs/isfusion/isfusion_0075voxel.py"
with open(filepath, "r") as f:
    code = f.read()

# 1. 极致缩减 Voxel 数量上限 (进一步砍到 15000/20000，对于训练测试完全足够跑通流程)
code = re.sub(r'max_voxels=\(\s*\d+,\s*\d+,\s*\)', 'max_voxels=(15000, 20000),', code)

# 2. 削减 3D 中间编码器的隐藏通道数 (从 [32, 64, 128, 256] 降到 [16, 32, 64, 128])
code = code.replace("base_channels=32", "base_channels=16")
code = re.sub(r'encoder_channels=\(\s*\(\s*32,\s*32,\s*64,\s*\),\s*\(\s*64,\s*64,\s*128,\s*\),\s*\(\s*128,\s*128,\s*256,\s*\),\s*\(\s*256,\s*256,\s*\),\s*\)', 
              'encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),', code)
code = code.replace("in_channels=64", "in_channels=32") # 针对 SparseEncoder 的输入
code = code.replace("output_channels=256", "output_channels=128") # 针对 SparseEncoder 的输出

# 3. 相应地修改后续 2D Backbone (SECONDV2) 的输入通道
code = re.sub(r"in_channels=128,\s*layer_nums=\[\s*5,\s*5,\s*\],\s*layer_strides=\[\s*1,\s*2,\s*\],\s*norm_cfg=dict\(eps=0\.001,\s*momentum=0\.01,\s*type='BN'\),\s*out_channels=\[\s*128,\s*256,\s*\]",
              "in_channels=128, layer_nums=[5, 5], layer_strides=[1, 2], norm_cfg=dict(eps=0.001, momentum=0.01, type='BN'), out_channels=[64, 128]", code)

# 4. 相应地修改 Neck (SECONDFPN) 的通道
code = re.sub(r"in_channels=\[\s*128,\s*256,\s*\]", "in_channels=[64, 128]", code)
code = re.sub(r"out_channels=\[\s*256,\s*256,\s*\]", "out_channels=[128, 128]", code)

# 5. 修改 TransFusionBBoxHead 的输入通道 (从 512 降到 256，因为 Neck 输出相加 128+128=256)
code = re.sub(r"in_channels=512,\s*loss_bbox=", "in_channels=256, loss_bbox=", code)

with open(filepath, "w") as f:
    f.write(code)

print("✅ 核武级显存压缩完成！Voxel数量和通道数均已大幅下调。")
