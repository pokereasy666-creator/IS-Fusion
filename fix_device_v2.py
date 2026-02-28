import re

filepath = "mmdet3d/models/middle_encoders/fusion_encoder.py"
with open(filepath, "r") as f:
    code = f.read()

# 替换掉错误的设备引用
old_line = "res = res.to(device=reference_points.device, dtype=reference_points.dtype)"
new_line = "res = res.to(device=img_features[0].device, dtype=img_features[0].dtype)"

if old_line in code:
    code = code.replace(old_line, new_line)
    print("✅ 找到并替换了错误的 reference_points 设备引用！")
else:
    print("⚠️ 没找到旧的设备引用，可能之前替换有误，请检查文件。")

with open(filepath, "w") as f:
    f.write(code)

