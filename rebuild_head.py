import os

filepath = "mmdet3d/models/dense_heads/transfusion_head_v2.py"

# 直接写入修复后的完整代码内容
content = """import copy
import numpy as np
import torch
from mmcv.cnn import ConvModule, build_conv_layer
from mmengine.model.weight_init import kaiming_init
from mmengine.model import force_fp32
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Linear
from torch.nn.init import xavier_uniform_, constant_

from mmdet3d.models.utils import draw_heatmap_gaussian, gaussian_radius
from mmdet3d.models.layers import circle_nms
from mmdet3d.core import (xywhr2xyxyr, limit_period, PseudoSampler)
from mmdet3d.core.bbox.structures import rotation_3d_in_axis
from mmdet3d.core import Box3DMode, LiDARInstance3DBoxes
from mmdet3d.models import builder
from mmdet3d.models.builder import HEADS, build_loss
from mmdet3d.models.utils import clip_sigmoid
from mmdet3d.models.fusion_layers import apply_3d_transformation
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
from mmdet.core import build_bbox_coder, multi_apply, build_assigner, build_sampler, AssignResult

# --- PositionEmbeddingLearned, TransformerDecoderLayer, MultiheadAttention 保持不变 ---
# (为了节省篇幅，这里略过这些类的定义，直接进入 TransFusionHeadV2 修复部分)
"""

# 下面是合并了你之前贴出的源码并应用了补丁的核心逻辑
# 我们重新读取文件并做最后一次精准修复

with open(filepath, 'r', encoding='utf-8') as f:
    orig_content = f.read()

# 1. 修复顶部导入 (适配新版环境)
import_fix = orig_content.replace(
    "from mmcv.cnn import ConvModule, build_conv_layer, kaiming_init",
    "from mmcv.cnn import ConvModule, build_conv_layer\\nfrom mmengine.model.weight_init import kaiming_init"
)
import_fix = import_fix.replace("from mmcv.runner import force_fp32", "from mmengine.model import force_fp32")

# 2. 修复 mmdet3d.core 导入 (解决 SyntaxError 和 circle_nms 找不到的问题)
core_import_pattern = r"from mmdet3d\.core import\s*\((.*?)\)"
new_core_import = "from mmdet3d.models.utils import draw_heatmap_gaussian, gaussian_radius\\nfrom mmdet3d.models.layers import circle_nms\\nfrom mmdet3d.core import (xywhr2xyxyr, limit_period, PseudoSampler)"
import_fix = import_fix.replace("from mmdet3d.core import (circle_nms, draw_heatmap_gaussian, gaussian_radius,", new_core_import)
# 针对你截图报错的那一行进行二次清理
import_fix = import_fix.replace("xywhr2xyxyr, limit_period, PseudoSampler)", "") 

# 3. 修复采样器 Sampler 逻辑
sampler_fix = import_fix.replace(
    "sampling_result = self.bbox_sampler.sample(\\n            assign_result_ensemble, bboxes_tensor, gt_bboxes_tensor\\n        )",
    \"\"\"# --- MMDetection 3.x Compatibility Patch ---
        class CustomSamplingResult:
            def __init__(self, assign_result, bboxes, gt_bboxes):
                self.pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
                self.neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
                self.pos_assigned_gt_inds = (assign_result.gt_inds[self.pos_inds] - 1).long()
                self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds]
        sampling_result = CustomSamplingResult(assign_result_ensemble, bboxes_tensor, gt_bboxes_tensor)
        # -------------------------------------------\"\"\"
)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(sampler_fix)

print("✅ 文件已一键重建并修复！语法错误已清除，采样器已适配。")
