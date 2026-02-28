import os
import re

# 1. 恢复配置文件中的 with_cp=True (开启梯度检查点，拯救显存)
conf = "configs/isfusion/isfusion_0075voxel_8gb.py"
if os.path.exists(conf):
    with open(conf, "r", encoding="utf-8") as f: code = f.read()
    code = re.sub(r'with_cp\s*=\s*False', 'with_cp=True', code)
    with open(conf, "w", encoding="utf-8") as f: f.write(code)

# 2. 修改 tools/train.py，强开 with_cp 并强关 find_unused_parameters
train_tool = "tools/train.py"
if os.path.exists(train_tool):
    with open(train_tool, "r", encoding="utf-8") as f: code = f.read()
    code = code.replace('cfg.model.img_backbone.with_cp = False', 'cfg.model.img_backbone.with_cp = True')
    if 'cfg.find_unused_parameters = False' not in code:
        code = code.replace('print("🚀 [HOTFIX]', 'cfg.find_unused_parameters = False\n    print("🚀 [HOTFIX]')
    with open(train_tool, "w", encoding="utf-8") as f: f.write(code)

# 3. 擦除之前我们在 mmdet3d/apis/train.py 中强加的 find_unused_parameters=True
train_api = "mmdet3d/apis/train.py"
if os.path.exists(train_api):
    with open(train_api, "r", encoding="utf-8") as f: code = f.read()
    code = re.sub(r'find_unused_parameters\s*=\s*True', 'find_unused_parameters=False', code)
    with open(train_api, "w", encoding="utf-8") as f: f.write(code)

print("✅ 【终极组合技】已开启 Checkpoint 节省显存，并彻底关闭 DDP 冲突检测！")
