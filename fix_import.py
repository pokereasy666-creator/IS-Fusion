import re

filepath = "mmdet3d/models/detectors/isfusion.py"
with open(filepath, "r") as f:
    code = f.read()

# 1. 强制清理掉刚才产生缩进错误的 try-except 块
bad_try_block = r"try:\s*from mmdet\.models import DETECTORS\s*except ImportError:\s*from mmdet\.registry import MODELS as DETECTORS"
code = re.sub(bad_try_block, "from mmdet.registry import MODELS as DETECTORS", code)

# 2. 替换任何可能残留的旧版导入为“单行流”（完全不需要缩进，100%安全）
code = code.replace("from mmdet.models import DETECTORS", "from mmdet.registry import MODELS as DETECTORS")
code = code.replace("from mmdet3d.models import DETECTORS", "from mmdet.registry import MODELS as DETECTORS")
code = code.replace("from mmdet3d.models.builder import DETECTORS", "from mmdet.registry import MODELS as DETECTORS")

with open(filepath, "w") as f:
    f.write(code)

print("✅ 注册表导入已替换为单行无缩进版本，彻底避开缩进错误！")
