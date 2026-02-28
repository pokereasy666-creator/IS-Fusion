import re

filepath = "mmdet3d/models/dense_heads/transfusion_head_v2.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# 匹配我们刚才不小心插错位置的补丁
pattern = r'# --- 强制补回 force_fp32 ---.*?# --------------------------\n?'
match = re.search(pattern, code, flags=re.DOTALL)

if match:
    block = match.group(0)
    # 1. 把它从破坏装饰器语法的地方彻底删掉
    code = code.replace(block, "")
    
    # 2. 把它安全地放到文件最上面（import torch 的下方）
    code = code.replace("import torch\n", "import torch\n" + block + "\n", 1)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    print("✅ 语法错误解除：force_fp32 补丁已移至安全区域，装饰器恢复正常！")
else:
    print("⚠️ 没找到需要移动的补丁块，请确认文件状态。")
