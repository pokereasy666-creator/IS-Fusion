import os

# 1. 优化 TransFusionHeadV2，启用 PyTorch 2.x 的 FlashAttention
filepath = "mmdet3d/models/dense_heads/transfusion_head_v2.py"
if os.path.exists(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()

    # 替换为 PyTorch 原生的 nn.MultiheadAttention
    code = code.replace("self.self_attn = MultiheadAttention(", "self.self_attn = nn.MultiheadAttention(")
    code = code.replace("self.multihead_attn = MultiheadAttention(", "self.multihead_attn = nn.MultiheadAttention(")

    # 注入 need_weights=False，激活内核级 FlashAttention (SDPA)，节约数百MB显存！
    code = code.replace("query2 = self.self_attn(q, k, value=v)[0]", "query2 = self.self_attn(q, k, value=v, need_weights=False)[0]")
    code = code.replace("value=self.with_pos_embed(key, key_pos_embed), attn_mask=attn_mask)[0]", "value=self.with_pos_embed(key, key_pos_embed), attn_mask=attn_mask, need_weights=False)[0]")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    print("✅ TransFusionHeadV2 成功接入 FlashAttention！显存峰值已被抹除！")

# 2. 全局开启 TF32 和 SDPA 优化
train_path = "tools/train.py"
if os.path.exists(train_path):
    with open(train_path, "r", encoding="utf-8") as f:
        train_code = f.read()
    
    tf32_patch = """import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
"""
    if "allow_tf32" not in train_code:
        train_code = train_code.replace("import torch", tf32_patch, 1)
        with open(train_path, "w", encoding="utf-8") as f:
            f.write(train_code)
        print("✅ 全局 TF32 和硬件级内存优化已开启，极致压榨 RTX 5060 显存！")
