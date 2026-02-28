filepath = "mmdet3d/models/dense_heads/transfusion_head_v2.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# 1. 补回不小心被吃掉的 PyTorch 基础导入（放在类的正上方最安全）
safe_imports = """
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Linear
from torch.nn.init import xavier_uniform_, constant_
"""
# 防止重复导入，先简单判断一下
if "class PositionEmbeddingLearned(nn.Module):" in code:
    code = code.replace(
        "class PositionEmbeddingLearned(nn.Module):", 
        safe_imports + "\nclass PositionEmbeddingLearned(nn.Module):"
    )

# 2. 给我们的虚拟模块补发 build_bbox_coder 的“假护照”
patch = """mcb.BaseBBoxCoder = BaseBBoxCoder
    
    # 骗过 build_bbox_coder 导入
    try:
        from mmdet.registry import MODELS
        mcb.build_bbox_coder = MODELS.build
    except ImportError:
        pass
"""
code = code.replace("mcb.BaseBBoxCoder = BaseBBoxCoder", patch)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)

print("✅ 缺失的 nn 已补回！")
print("✅ 虚拟模块已升级，成功拦截 build_bbox_coder！")
