import re

filepath = "mmdet3d/models/dense_heads/transfusion_head_v2.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# 1. 匹配那一整块物理缝合包
fusion_pattern = r'# =====================================================================\n# ☢️ ULTIMATE PHYSICAL FUSION.*?# =====================================================================\n'
match = re.search(fusion_pattern, code, flags=re.DOTALL)

if match:
    fusion_block = match.group(0)
    
    # 2. 从错误位置彻底抹除它
    code = code.replace(fusion_block, "")
    
    # 3. 修复被劈开的装饰器（确保装饰器下面紧跟着 class）
    code = re.sub(r'@HEADS\.register_module\(\)\s*(?:\n\s*)*class TransFusionHeadV2', '@HEADS.register_module()\nclass TransFusionHeadV2', code)
    
    # 4. 把缝合包放在整个文件的【最前面】（绝对安全区）
    code = fusion_block + "\n" + code
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    print("✅ 缝合包已被强制移至文件最顶端（第1行），彻底摆脱装饰器干扰！")
else:
    print("⚠️ 没找到缝合块，文件可能已被修改。")
