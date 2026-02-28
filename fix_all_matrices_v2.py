import re

filepath = "mmdet3d/models/middle_encoders/fusion_encoder.py"
with open(filepath, "r") as f:
    code = f.read()

# 这一段正则表达式匹配形如: 
# cur_X = X[b] if not isinstance(X, list) else X[0][b]
# 这种 MMDetection3D 常用的丑陋解包代码，将其全部替换为安全解包
pattern = r"([a-zA-Z0-9_]+)\s*=\s*([a-zA-Z0-9_]+)\[b\] if not isinstance\(\2, list\) else \2\[0\]\[b\]"

def replacer(match):
    target_var = match.group(1)
    source_var = match.group(2)
    return f"""
            # 安全解包 {source_var}
            val = {source_var}
            while isinstance(val, list) and len(val) == 1: val = val[0]
            if hasattr(val, 'data'): val = val.data
            while isinstance(val, list) and len(val) == 1: val = val[0]
            if hasattr(val, 'data'): val = val.data
            
            if isinstance(val, list) or isinstance(val, torch.Tensor):
                {target_var} = val[b]
            else:
                {target_var} = val
    """

# 执行全局替换
new_code = re.sub(pattern, replacer, code)

# 确保在函数开头有我们之前的 _get_val 方法定义，并且它拿出了 lidar2image
extract_block_target = "image_size = _get_val('input_shape')"
if "lidar2image = _get_val('lidar2image')" not in new_code:
    extra_extracts = """
        lidar2image = _get_val('lidar2image')
        if isinstance(lidar2image, torch.Tensor): lidar2image = [lidar2image]
"""
    new_code = new_code.replace(extract_block_target, extract_block_target + extra_extracts)

with open(filepath, "w") as f:
    f.write(new_code)

print("✅ fix_all_matrices_v2.py 执行成功：所有可能的矩阵变量已被彻底降维打击！")
