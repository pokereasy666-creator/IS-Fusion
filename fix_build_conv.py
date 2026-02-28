filepath = "mmdet3d/models/dense_heads/transfusion_head_v2.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# 检查是否已经导入了
if "from mmcv.cnn import ConvModule, build_conv_layer" not in code:
    # 极其安全地在 import numpy 的下方插入
    code = code.replace("import numpy as np\n", "import numpy as np\nfrom mmcv.cnn import ConvModule, build_conv_layer\n", 1)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    print("✅ 成功补入 ConvModule 和 build_conv_layer！")
else:
    print("⚠️ 已经存在导入。")
