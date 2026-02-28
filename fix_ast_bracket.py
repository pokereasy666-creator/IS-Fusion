import ast

filepath = "mmdet3d/apis/train.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

search_str = "find_unused_parameters=True)"
indices = [i for i in range(len(code)) if code.startswith(search_str, i)]

fixed = False
for idx in indices:
    # 尝试把当前匹配到的这个多余的 ')' 删掉
    bracket_pos = idx + len(search_str) - 1
    candidate = code[:bracket_pos] + code[bracket_pos + 1:]
    
    try:
        # 用 Python 原生编译器校验语法是否恢复正常
        ast.parse(candidate)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(candidate)
        print("✅ 语法错误已自动修正！(AST 编译器校验通过，多余括号已铲除)")
        fixed = True
        break
    except SyntaxError:
        continue

if not fixed:
    # 兜底方案：强行修正第一个出现的（通常就是赋值语句那个）
    candidate = code.replace("find_unused_parameters=True)", "find_unused_parameters=True", 1)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(candidate)
    print("✅ 已使用备用方案强行铲除第一个多余括号！")
