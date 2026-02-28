import os
import re

# 1. 强制禁用 benchmark，释放几百MB的隐藏占用
train_file = "tools/train.py"
if os.path.exists(train_file):
    with open(train_file, "r", encoding="utf-8") as f:
        code = f.read()
    code = code.replace("cudnn.benchmark = True", "cudnn.benchmark = False")
    code = re.sub(r"cfg\.cudnn_benchmark\s*=\s*True", "cfg.cudnn_benchmark = False", code)
    if "torch.backends.cudnn.benchmark = False" not in code:
        code = code.replace("import torch", "import torch\ntorch.backends.cudnn.benchmark = False")
    with open(train_file, "w", encoding="utf-8") as f:
        f.write(code)

# 2. 定点爆破：在 SECOND FPN 的高危 contiguous() 前一秒强行回收显存
fpn_file = "mmdet3d/models/necks/second_fpn.py"
if os.path.exists(fpn_file):
    with open(fpn_file, "r", encoding="utf-8") as f:
        fpn_code = f.read()
    old_str = "out = out.permute(0, 1, 3, 2).contiguous()"
    new_str = "del ups\n        import gc, torch\n        gc.collect()\n        torch.cuda.empty_cache()\n        out = out.permute(0, 1, 3, 2).contiguous()"
    if "del ups" not in fpn_code:
        fpn_code = fpn_code.replace(old_str, new_str)
        with open(fpn_file, "w", encoding="utf-8") as f:
            f.write(fpn_code)

print("✅ 极限榨汁完成：白嫖数百MB工作区，定点内存回收已部署！")
