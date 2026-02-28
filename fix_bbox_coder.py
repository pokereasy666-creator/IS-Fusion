filepath = "mmdet3d/models/dense_heads/transfusion_head_v2.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

force_import_code = """
# --- 强制注册 TransFusionBBoxCoder 补丁 ---
try:
    from mmdet3d.core.bbox.coders.transfusion_bbox_coder import TransFusionBBoxCoder
    from mmdet.registry import TASK_UTILS, MODELS
    if 'TransFusionBBoxCoder' not in TASK_UTILS:
        TASK_UTILS.register_module(module=TransFusionBBoxCoder, force=True)
    if 'TransFusionBBoxCoder' not in MODELS:
        MODELS.register_module(module=TransFusionBBoxCoder, force=True)
except ImportError:
    pass
# -------------------------------------------
"""

# 将这段代码安全地放在 class 定义之前
if "强制注册 TransFusionBBoxCoder" not in code:
    code = code.replace("class PositionEmbeddingLearned", force_import_code + "\nclass PositionEmbeddingLearned")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    print("✅ 已成功注入 TransFusionBBoxCoder 强制注册逻辑！")
else:
    print("⚠️ 已经注入过了。")
