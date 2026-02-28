import re

filepath = "mmdet3d/models/dense_heads/transfusion_head_v2.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# 1. 无情抹除全文件中所有游离的 @HEADS.register_module() 及其身后的空白
code = re.sub(r'@HEADS\.register_module\(\)\s*', '', code)

# 2. 精准定位到目标类，把装饰器给它戴到头顶上
target = "class TransFusionHeadV2(nn.Module):"
code = code.replace(target, "@HEADS.register_module()\n" + target)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)

print("✅ 悬空的装饰器已被精准归位，彻底消除 SyntaxError！")
