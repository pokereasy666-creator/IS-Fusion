import os

filepath = "tools/train.py"
with open(filepath, "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "torch.backends.cuda.matmul.allow_tf32 = True" in line:
        # 自动提取上一行 (import torch) 的原生缩进
        indent = ""
        if i > 0 and "import torch" in lines[i-1]:
            indent = lines[i-1][:len(lines[i-1]) - len(lines[i-1].lstrip())]
        
        # 将相同的缩进赋予我们注入的优化代码
        lines[i] = indent + "torch.backends.cuda.matmul.allow_tf32 = True\n"
        if i+1 < len(lines) and "torch.backends.cudnn.allow_tf32" in lines[i+1]:
            lines[i+1] = indent + "torch.backends.cudnn.allow_tf32 = True\n"
        if i+2 < len(lines) and "enable_flash_sdp" in lines[i+2]:
            lines[i+2] = indent + "torch.backends.cuda.enable_flash_sdp(True)\n"
        if i+3 < len(lines) and "enable_mem_efficient_sdp" in lines[i+3]:
            lines[i+3] = indent + "torch.backends.cuda.enable_mem_efficient_sdp(True)\n"
        break

with open(filepath, "w", encoding="utf-8") as f:
    f.writelines(lines)

print("✅ 缩进错误已完美修复！")
