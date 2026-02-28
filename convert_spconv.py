import torch

ckpt_path = 'checkpoints/IS-Fusion_epoch_10.pth'
out_path = 'checkpoints/IS-Fusion_epoch_10_spconv2.pth'

print(f"Loading old checkpoint from {ckpt_path}...")
ckpt = torch.load(ckpt_path, map_location='cpu')
state_dict = ckpt['state_dict']

for k, v in state_dict.items():
    # 找到 3D 中间编码器里的 5 维稀疏卷积权重
    if 'pts_middle_encoder' in k and len(v.shape) == 5:
        # spconv 1.x: [out_channels, D, H, W, in_channels]
        # spconv 2.x: [D, H, W, in_channels, out_channels]
        # 通过 permute 把 第0维度(out) 挪到最后
        new_v = v.permute(1, 2, 3, 4, 0).contiguous()
        state_dict[k] = new_v
        print(f"[Converted] {k}: {v.shape} -> {new_v.shape}")

ckpt['state_dict'] = state_dict
torch.save(ckpt, out_path)
print(f"\n[Success] Converted checkpoint saved to {out_path}!")
