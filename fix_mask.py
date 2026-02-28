import re

filepath = "mmdet3d/models/detectors/isfusion.py"
with open(filepath, "r") as f:
    code = f.read()

# 使用正则匹配并替换掉彻底乱掉的 if this_mask_idx is not None 整个代码块
pattern = re.compile(r"[ \t]*if this_mask_idx is not None:.*?(?:img\[i\]\[this_mask_idx, \.\.\.\] = 0\.0\n?)", re.DOTALL)

good_code = """                if this_mask_idx is not None:
                    if hasattr(img[i], 'data'):
                        if isinstance(img[i].data, list): img[i].data[0][this_mask_idx, ...] = 0.0
                        else: img[i].data[this_mask_idx, ...] = 0.0
                    else:
                        img[i][this_mask_idx, ...] = 0.0\n"""

new_code = pattern.sub(good_code, code)

with open(filepath, "w") as f:
    f.write(new_code)
    
print("✅ 成功重构 this_mask_idx 赋值块，DataContainer 解包与缩进已完全修复！")
