import re

filepath = "mmdet3d/models/middle_encoders/fusion_encoder.py"
with open(filepath, "r") as f:
    code = f.read()

# 替换掉昨天那个找不到变量的设备引用
old_line = "res = res.to(device=img_features[0].device, dtype=img_features[0].dtype)"
new_line = "res = res.cuda().float()"

if old_line in code:
    code = code.replace(old_line, new_line)
    print("✅ 找到并替换了错误的 img_features 引用，现已使用绝对 .cuda() 转移！")
else:
    print("⚠️ 没找到旧的设备引用，请检查文件。")

with open(filepath, "w") as f:
    f.write(code)

