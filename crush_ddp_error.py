import os
import re

# 1. 强制在训练脚本中开启 find_unused_parameters=True
train_api = "mmdet3d/apis/train.py"
if os.path.exists(train_api):
    with open(train_api, "r", encoding="utf-8") as f:
        code = f.read()
    
    code = re.sub(r'find_unused_parameters\s*=\s*[^,\)]+', 'find_unused_parameters=True', code)
    
    with open(train_api, "w", encoding="utf-8") as f:
        f.write(code)
    print("✅ DDP 引擎配置已强制开启：允许未使用参数 (find_unused_parameters=True)！")

# 2. 在模型底层添加一个权重为 0 的 Dummy Loss（物理级绝对防御）
model_file = "mmdet3d/models/detectors/isfusion.py"
if os.path.exists(model_file):
    with open(model_file, "r", encoding="utf-8") as f:
        code = f.read()
    
    dummy_patch = """# --- 终极保底：Dummy Loss 确保 100% 参数进入计算图 ---
        dummy_loss = 0.0
        for p in self.parameters():
            if p.requires_grad:
                dummy_loss = dummy_loss + p.sum() * 0.0
        if isinstance(losses, dict):
            losses['loss_dummy_safeguard'] = dummy_loss
        return losses"""
    
    if "loss_dummy_safeguard" not in code:
        code = re.sub(r'return losses\s*$', dummy_patch, code, flags=re.MULTILINE)
        with open(model_file, "w", encoding="utf-8") as f:
            f.write(code)
        print("✅ Dummy Loss 物理护盾已加载！所有参数都会完美接收梯度同步！")
else:
    print("⚠️ 未找到检测器主文件，请确认路径。")
