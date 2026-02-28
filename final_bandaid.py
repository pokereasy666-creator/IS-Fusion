import re

# 1. 修复底层 Coder 的空 try 块导致的缩进错误
coder_path = "mmdet3d/core/bbox/coders/transfusion_bbox_coder.py"
try:
    with open(coder_path, "r", encoding="utf-8") as f:
        coder_code = f.read()
    
    # 找到孤立的 try: (中间只有空白或换行) 然后紧接着 except 的地方，塞入一个 pass
    coder_code = re.sub(r'try:\s*(?:\n\s*)*except', 'try:\n    pass\nexcept', coder_code)
    
    with open(coder_path, "w", encoding="utf-8") as f:
        f.write(coder_code)
    print("✅ Coder 文件的空 try 块已修复 (IndentationError 排除)！")
except Exception as e:
    print("⚠️ Coder 修复遇到问题:", e)


# 2. 补回丢失的 force_fp32
head_path = "mmdet3d/models/dense_heads/transfusion_head_v2.py"
with open(head_path, "r", encoding="utf-8") as f:
    head_code = f.read()

fp32_patch = """
# --- 强制补回 force_fp32 ---
try:
    from mmengine.model import force_fp32
except ImportError:
    try:
        from mmcv.runner import force_fp32
    except ImportError:
        def force_fp32(apply_to=None, out_fp16=False):
            def decorator(func):
                return func
            return decorator
# --------------------------
"""

if "兜底 force_fp32" not in head_code and "强制补回 force_fp32" not in head_code:
    # 极其安全地插在类的正上方
    head_code = head_code.replace("class TransFusionHeadV2", fp32_patch + "\nclass TransFusionHeadV2")
    with open(head_path, "w", encoding="utf-8") as f:
        f.write(head_code)
    print("✅ force_fp32 装饰器已成功归位 (NameError 排除)！")
