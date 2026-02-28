import os
import re

filepath = "tools/train.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

patch = """
# --- 🚀 HOTFIX: 彻底解决 PyTorch 2.x DDP 与 Checkpoint 冲突 ---
import torch
import torch.utils.checkpoint
_orig_cp = torch.utils.checkpoint.checkpoint

def _patched_cp(*args, **kwargs):
    kwargs['use_reentrant'] = False
    return _orig_cp(*args, **kwargs)

torch.utils.checkpoint.checkpoint = _patched_cp
print("🚀 [HOTFIX] 已强制全局 Checkpoint 使用 use_reentrant=False！")
# -------------------------------------------------------------
"""

if "use_reentrant=False" not in code:
    # 找到 import argparse 的地方，在它前面注入劫持代码
    code = re.sub(r'(import argparse)', patch + r'\n\1', code, count=1)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    print("✅ 全局函数劫持成功！PyTorch 将安全执行梯度检查点！")
else:
    print("⚠️ 补丁已存在。")
