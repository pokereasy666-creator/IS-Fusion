import re
import os

head_path = "mmdet3d/models/dense_heads/transfusion_head_v2.py"
coder_path = "mmdet3d/core/bbox/coders/transfusion_bbox_coder.py"

# 1. 提取并净化 Coder 源码
if os.path.exists(coder_path):
    with open(coder_path, "r", encoding="utf-8") as f:
        coder_code = f.read()
    
    # 提取 class TransFusionBBoxCoder(BaseBBoxCoder): 到文件末尾的所有内容
    match = re.search(r'class TransFusionBBoxCoder.*', coder_code, flags=re.DOTALL)
    if match:
        pure_coder_class = match.group(0)
        
        # 净化：把继承的 BaseBBoxCoder 改为 object（彻底摆脱对 mmdet.core 的依赖）
        pure_coder_class = pure_coder_class.replace("class TransFusionBBoxCoder(BaseBBoxCoder):", "class TransFusionBBoxCoder(object):")
        
        # 构造终极缝合包
        injection = f"""
# =====================================================================
# ☢️ ULTIMATE PHYSICAL FUSION: TransFusionBBoxCoder INLINED ☢️
# =====================================================================
import torch
import numpy as np

# 直接在本地免注册使用
{pure_coder_class}

# 强行劫持 build_bbox_coder，让它直接返回我们的本地类！
def custom_build_bbox_coder(cfg, **kwargs):
    if cfg.get('type') == 'TransFusionBBoxCoder':
        # 剔除 type 字段
        cfg_copy = cfg.copy()
        cfg_copy.pop('type')
        return TransFusionBBoxCoder(**cfg_copy)
    else:
        # 其他的 coder 交给默认注册器
        from mmdet.registry import MODELS
        return MODELS.build(cfg)

build_bbox_coder = custom_build_bbox_coder
# =====================================================================
"""
        
        # 2. 将缝合包注入到 Head 文件的类定义之前
        with open(head_path, "r", encoding="utf-8") as f:
            head_code = f.read()
            
        # 先清理掉我们之前尝试的所有失败的强制注册逻辑
        head_code = re.sub(r'# --- MMDetection 3\.x System-wide Mock for Coder ---.*?# --------------------------------------------------', '', head_code, flags=re.DOTALL)
        
        if "ULTIMATE PHYSICAL FUSION" not in head_code:
            head_code = head_code.replace("class TransFusionHeadV2", injection + "\nclass TransFusionHeadV2")
            with open(head_path, "w", encoding="utf-8") as f:
                f.write(head_code)
            print("✅ 终极物理缝合完成！BBox Coder 已被完全吸收！注册器已被强行接管！")
        else:
            print("⚠️ 已经缝合过了。")
    else:
        print("❌ 提取 Coder 类失败。")
else:
    print(f"❌ 找不到文件: {coder_path}")
