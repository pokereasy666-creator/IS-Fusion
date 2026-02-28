import re

filepath = "mmdet3d/models/dense_heads/transfusion_head_v2.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# 1. 匹配出那一整块插错位置的物理缝合包
fusion_pattern = r'# =====================================================================\n# ☢️ ULTIMATE PHYSICAL FUSION.*?# =====================================================================\n'
match = re.search(fusion_pattern, code, flags=re.DOTALL)

if match:
    fusion_block = match.group(0)
    
    # 2. 从错误位置（破坏装饰器的地方）把它彻底删掉
    code = code.replace(fusion_block, "")
    
    # 3. 修复可能因为删除而产生的多余空行，确保装饰器和类紧紧贴在一起
    code = re.sub(r'@HEADS\.register_module\(\)\s+class TransFusionHeadV2', '@HEADS.register_module()\nclass TransFusionHeadV2', code)
    
    # 4. 把缝合包安全地放置在第一个类（PositionEmbeddingLearned）的上方
    code = code.replace("class PositionEmbeddingLearned", fusion_block + "\nclass PositionEmbeddingLearned")
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    print("✅ 缝合包已成功移出装饰器雷区，语法错误解除！")
else:
    print("⚠️ 未找到缝合代码块，请确认文件状态。")
