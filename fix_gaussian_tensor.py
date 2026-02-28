import re

filepath = "mmdet3d/models/dense_heads/transfusion_head_v2.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

old_func = r'def draw_heatmap_gaussian\(heatmap, center, radius, k=1\):.*?return heatmap'

new_func = """def draw_heatmap_gaussian(heatmap, center, radius, k=1):
    radius = int(radius)
    diameter = 2 * radius + 1
    
    # 纯 PyTorch 实现，完全脱离 Numpy，速度更快且无兼容性问题
    x = torch.arange(-radius, radius + 1, dtype=heatmap.dtype, device=heatmap.device).view(1, -1)
    y = torch.arange(-radius, radius + 1, dtype=heatmap.dtype, device=heatmap.device).view(-1, 1)
    
    h = torch.exp(-(x ** 2 + y ** 2) / (2 * (diameter / 6) ** 2))
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    
    x_c, y_c = int(center[0]), int(center[1])
    
    left, top = min(x_c, radius), min(y_c, radius)
    right, bottom = min(heatmap.shape[1] - x_c, radius + 1), min(heatmap.shape[0] - y_c, radius + 1)
    
    masked_heatmap  = heatmap[y_c - top:y_c + bottom, x_c - left:x_c + right]
    masked_gaussian = h[radius - top:radius + bottom, radius - left:radius + right]
    
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap"""

code = re.sub(old_func, new_func, code, flags=re.DOTALL)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)

print("✅ 高斯热图函数已重写为纯 PyTorch 版本！")
