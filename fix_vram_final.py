import os

filepath = "tools/train.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# 1. 把刚才关闭的梯度检查点（显存救星）重新开启！
code = code.replace("cfg.model.img_backbone.with_cp = False", "cfg.model.img_backbone.with_cp = True")

# 2. 强行关闭 find_unused_parameters，彻底化解 Checkpoint 与 DDP 的深层冲突！
code = code.replace(
    'print("🚀 [HOTFIX] Checkpointing disabled at runtime!")', 
    'cfg.find_unused_parameters = False\n    print("🚀 [HOTFIX] 终极护盾启动: with_cp=True (省显存), find_unused=False (解冲突)!")'
)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)

print("✅ 终极显存与梯度护盾已就绪！")
