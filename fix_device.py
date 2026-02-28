import re

filepath = "mmdet3d/models/middle_encoders/fusion_encoder.py"
with open(filepath, "r") as f:
    code = f.read()

# 定位 _get_item 函数并替换它
old_func = """            def _get_item(matrix_var, batch_idx):
                val = matrix_var
                while isinstance(val, list) and len(val) == 1: val = val[0]
                if hasattr(val, 'data'): val = val.data
                while isinstance(val, list) and len(val) == 1: val = val[0]
                if hasattr(val, 'data'): val = val.data
                if isinstance(val, list) or isinstance(val, torch.Tensor):
                    return val[batch_idx]
                return val"""

new_func = """            def _get_item(matrix_var, batch_idx):
                val = matrix_var
                while isinstance(val, list) and len(val) == 1: val = val[0]
                if hasattr(val, 'data'): val = val.data
                while isinstance(val, list) and len(val) == 1: val = val[0]
                if hasattr(val, 'data'): val = val.data
                
                res = val[batch_idx] if (isinstance(val, list) or isinstance(val, torch.Tensor)) else val
                
                # 自动将取出来的 Tensor 送到 GPU
                if isinstance(res, torch.Tensor):
                    res = res.to(device=reference_points.device, dtype=reference_points.dtype)
                return res"""

if old_func in code:
    code = code.replace(old_func, new_func)
else:
    # 兼容 v2 版本的替换
    old_v2 = """            # 安全解包 {source_var}"""
    code = re.sub(
        r"(def _get_item\(matrix_var, batch_idx\):.*?return val)", 
        new_func.strip(), 
        code, 
        flags=re.DOTALL
    )

with open(filepath, "w") as f:
    f.write(code)

print("✅ fix_device.py 执行成功：提取出的变换矩阵将自动传送到 GPU！")
