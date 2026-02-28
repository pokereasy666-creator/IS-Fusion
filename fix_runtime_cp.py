import re

filepath = "tools/train.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# 精准定位配置加载的代码，动态注入 with_cp = False
pattern = r'^(\s*)(cfg\s*=\s*Config\.fromfile[^\n]+)'
replacement = r'\1\2\n\1if hasattr(cfg, "model") and "img_backbone" in cfg.model:\n\1    cfg.model.img_backbone.with_cp = False\n\1    print("🚀 [HOTFIX] Checkpointing disabled at runtime!")'

if "HOTFIX" not in code:
    code = re.sub(pattern, replacement, code, count=1, flags=re.MULTILINE)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    print("✅ 运行时配置拦截已注入，成功覆盖父级 config 中的 with_cp 设定！")
else:
    print("⚠️ 拦截代码已存在。")
