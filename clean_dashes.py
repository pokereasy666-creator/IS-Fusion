import re

filepath = "mmdet3d/models/dense_heads/transfusion_head_v2.py"
with open(filepath, "r", encoding="utf-8") as f:
    lines = f.readlines()

cleaned_lines = []
for i, line in enumerate(lines):
    # 匹配并忽略那些只有减号/破折号的行
    if re.match(r'^\s*-+\s*$', line):
        print(f"🧹 已清理第 {i+1} 行的乱码: {line.strip()}")
        continue
    cleaned_lines.append(line)

with open(filepath, "w", encoding="utf-8") as f:
    f.writelines(cleaned_lines)

print("✅ 语法错误清理完毕！")
