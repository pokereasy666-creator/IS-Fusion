import re

filepath = "mmdet3d/models/dense_heads/transfusion_head_v2.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# 报错的导入块
old_import = """try:
    from mmdet3d.models.layers import circle_nms
except ImportError:
    from mmdet3d.ops.iou3d.iou3d_utils import circle_nms"""

# 我们自己用纯 numpy 手写的、完全兼容的 circle_nms 算法！
inline_circle_nms = """
# --- [FIX 3] 内联 circle_nms (彻底解决 ImportError) ---
def circle_nms(dets, thresh):
    import numpy as np
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    scores = dets[:, 2]
    order = scores.argsort()[::-1].astype(np.int32)
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            dist = np.sqrt((x1[i] - x1[j])**2 + (y1[i] - y1[j])**2)
            if dist <= thresh:
                suppressed[j] = 1
    return keep
# -----------------------------------------------------
"""

if old_import in code:
    code = code.replace(old_import, inline_circle_nms)
else:
    # 兼容处理，万一缩进不一样
    code = re.sub(r'try:\s*from mmdet3d\.models\.layers import circle_nms\s*except ImportError:\s*from mmdet3d\.ops\.iou3d\.iou3d_utils import circle_nms', inline_circle_nms, code)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)

print("✅ 成功将 circle_nms 内联到代码中！所有外部依赖均已斩断！")
