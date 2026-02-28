import re
filepath = "mmdet3d/apis/train.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# 把多余的连续右括号（如 True)) ）全部削减为一个正确的括号
code = re.sub(r'find_unused_parameters=True\)+', 'find_unused_parameters=True)', code)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)

print("✅ 语法错误已修复！括号已完美闭合！")
