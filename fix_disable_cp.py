import os
import re

filepath = "configs/isfusion/isfusion_0075voxel_8gb.py"
if os.path.exists(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()
    
    # 将所有的 with_cp=True 替换为 with_cp=False
    code = re.sub(r'with_cp\s*=\s*True', 'with_cp=False', code)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    print("✅ 已成功在配置文件中关闭梯度检查点 (with_cp=False)！")
else:
    print("❌ 找不到文件:", filepath)
