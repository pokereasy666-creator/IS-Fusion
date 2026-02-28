import os
import re

file_path = "mmdet3d/models/dense_heads/transfusion_head_v2.py"
with open(file_path, "r", encoding="utf-8") as f:
    code = f.read()

# 定位到直接返回裸 Tensor 的地方
pattern = re.compile(r'res\s*=\s*\[\s*\[\s*rets\[0\]\[0\]\["bboxes"\],\s*rets\[0\]\[0\]\["scores"\],\s*rets\[0\]\[0\]\["labels"\]\.int\(\),?\s*\]\s*\]', re.DOTALL)

# 替换为标准的 LiDARInstance3DBoxes 包装
replacement = """try:
            from mmdet3d.core.bbox import LiDARInstance3DBoxes
        except ImportError:
            try:
                from mmdet3d.structures import LiDARInstance3DBoxes
            except ImportError:
                from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes
            
        raw_boxes = rets[0][0]["bboxes"]
        # 获取 box 的维度 (通常是 9)
        b_dim = raw_boxes.shape[-1] if raw_boxes.shape[0] > 0 else 9
        # TransFusion 的中心点预测即为物理重心，因此 origin=(0.5, 0.5, 0.5)
        wrapped_boxes = LiDARInstance3DBoxes(raw_boxes, box_dim=b_dim, origin=(0.5, 0.5, 0.5))
        
        res = [
            [
                wrapped_boxes,
                rets[0][0]["scores"],
                rets[0][0]["labels"].int(),
            ]
        ]"""

new_code = pattern.sub(replacement, code)

if new_code != code:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_code)
    print("✅ 成功修复：已将测试输出的 Tensor 包装为 LiDARInstance3DBoxes 对象！")
else:
    print("⚠️ 未找到匹配的代码段，请检查 transfusion_head_v2.py 文件结尾处逻辑。")
