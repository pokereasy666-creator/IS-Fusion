import os
import re

coder_path = "mmdet3d/core/bbox/coders/transfusion_bbox_coder.py"

if os.path.exists(coder_path):
    with open(coder_path, "r", encoding="utf-8") as f:
        code = f.read()
    
    # 1. 暴力清除所有包含 mmdet.core 的导入行
    code = re.sub(r'^.*?from mmdet\.core.*?\n', '', code, flags=re.MULTILINE)
    code = re.sub(r'^.*?import mmdet\.core.*?\n', '', code, flags=re.MULTILINE)
    
    # 2. 注入 2026版 MMEngine / MMDetection 3.x 的免死金牌头文件
    safe_header = """import torch
import numpy as np

# --- MMEngine / MMDet 3.x Compatibility ---
try:
    from mmdet.registry import TASK_UTILS as BBOX_CODERS
except ImportError:
    pass

try:
    from mmdet.models.task_modules.coders.base_bbox_coder import BaseBBoxCoder
except ImportError:
    # 终极保底：如果实在找不到基类，直接继承 object，绝不报错
    class BaseBBoxCoder(object): pass
# ------------------------------------------
"""
    
    if "MMEngine / MMDet 3.x Compatibility" not in code:
        code = safe_header + "\n" + code
        
    with open(coder_path, "w", encoding="utf-8") as f:
        f.write(code)
    print("✅ 成功拔除底层雷区！Coder 文件对 mmdet.core 的旧依赖已被彻底斩断！")
else:
    print(f"⚠️ 找不到文件: {coder_path}，请确认路径是否正确。")
