import os
import re

# 1. 寻找并修复 transfusion_bbox_coder.py 内部的老旧导入
coder_path = None
for root, dirs, files in os.walk('mmdet3d'):
    if 'transfusion_bbox_coder.py' in files:
        coder_path = os.path.join(root, 'transfusion_bbox_coder.py')
        break

if coder_path:
    with open(coder_path, 'r', encoding='utf-8') as f:
        coder_code = f.read()
    
    # 修复 BaseBBoxCoder 导入路径
    coder_code = re.sub(
        r'from mmdet\.core\.bbox.*? import BaseBBoxCoder',
        'try:\n    from mmdet.core.bbox.coders.base_bbox_coder import BaseBBoxCoder\nexcept ImportError:\n    from mmdet.models.task_modules.coders.base_bbox_coder import BaseBBoxCoder',
        coder_code
    )
    # 修复 BBOX_CODERS 注册器导入路径
    coder_code = re.sub(
        r'from mmdet\.core\.bbox\.builder import BBOX_CODERS',
        'try:\n    from mmdet.core.bbox.builder import BBOX_CODERS\nexcept ImportError:\n    from mmdet.registry import TASK_UTILS as BBOX_CODERS',
        coder_code
    )
    
    with open(coder_path, 'w', encoding='utf-8') as f:
        f.write(coder_code)
    print(f"✅ 成功修复底层文件: {coder_path}")
else:
    print("⚠️ 找不到 transfusion_bbox_coder.py，可能路径特殊。")

# 2. 将其显式注册到 transfusion_head_v2.py 中
head_path = "mmdet3d/models/dense_heads/transfusion_head_v2.py"
with open(head_path, "r", encoding="utf-8") as f:
    head_code = f.read()

inject_code = """
# --- 强制注册 TransFusionBBoxCoder ---
import sys
try:
    from mmdet3d.core.bbox.coders.transfusion_bbox_coder import TransFusionBBoxCoder
    from mmdet.registry import MODELS, TASK_UTILS
    if 'TransFusionBBoxCoder' not in MODELS:
        MODELS.register_module(module=TransFusionBBoxCoder, force=True)
    if 'TransFusionBBoxCoder' not in TASK_UTILS:
        TASK_UTILS.register_module(module=TransFusionBBoxCoder, force=True)
except Exception as e:
    print("⚠️ BBox Coder 导入失败，原因:", e)
# -------------------------------------
"""

if "# --- 强制注册 TransFusionBBoxCoder ---" not in head_code:
    head_code = head_code.replace("class TransFusionHeadV2", inject_code + "\nclass TransFusionHeadV2")
    with open(head_path, "w", encoding="utf-8") as f:
        f.write(head_code)
    print("✅ 成功在 Head 中注入 Coder 强制注册逻辑！")
