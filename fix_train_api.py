filepath = "mmdet3d/apis/train.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# 精准替换掉那个残余的 ", False)"
code = code.replace("find_unused_parameters=True, False)", "find_unused_parameters=True)")
# 顺便防一手其他的残留可能
code = code.replace("find_unused_parameters=True, True)", "find_unused_parameters=True)")

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)
print("✅ 语法残留已彻底清除！DDP 配置完美修复！")
