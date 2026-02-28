filepath = "mmdet3d/models/dense_heads/transfusion_head_v2.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

funcs = """
# --- Gaussian Heatmap Utils ---
def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    a1, b1, c1 = 1, (height + width), width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2
    a2, b2, c2 = 4, 2 * (height + width), (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2
    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (height + width), (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

def draw_heatmap_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = heatmap.new_tensor(np.ogrid[-radius:radius + 1, -radius:radius + 1])
    x, y = gaussian
    h = torch.exp(-(x ** 2 + y ** 2) / (2 * (diameter / 6) ** 2))
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    left, top = min(center[0], radius), min(center[1], radius)
    right, bottom = min(heatmap.shape[1] - center[0], radius + 1), min(heatmap.shape[0] - center[1], radius + 1)
    masked_heatmap  = heatmap[center[1] - top:center[1] + bottom, center[0] - left:center[0] + right]
    masked_gaussian = h[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap
# ------------------------------
"""

if "def gaussian_radius" not in code:
    code = code.replace("import numpy as np\n", "import numpy as np\n" + funcs + "\n", 1)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    print("✅ 高斯热图辅助函数已成功补回！")
else:
    print("⚠️ 已经存在该函数。")
