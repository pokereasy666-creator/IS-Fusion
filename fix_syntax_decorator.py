import re

filepath = "mmdet3d/models/dense_heads/transfusion_head_v2.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# 找到那段引起报错的代码块
pattern = r'# --- 强制注册 TransFusionBBoxCoder ---.*?# -------------------------------------'
match = re.search(pattern, code, flags=re.DOTALL)

if match:
    block = match.group(0)
    # 1. 把它从原来破坏装饰器语法的位置彻底删掉
    code = re.sub(pattern, "", code, flags=re.DOTALL)
    # 2. 把它安全地放到文件开头的 import torch 下方
    code = code.replace("import torch", "import torch\n" + block + "\n", 1)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    print("✅ 成功修复：已将注册逻辑移出装饰器区域，语法错误解除！")
else:
    print("⚠️ 未找到匹配的块，请确认文件状态。")
