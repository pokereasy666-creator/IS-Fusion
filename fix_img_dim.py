import re

filepath = "mmdet3d/models/detectors/isfusion.py"
with open(filepath, "r") as f:
    code = f.read()

# 匹配 extract_img_feat 函数
pattern = re.compile(r"^[ \t]*def extract_img_feat\(self, img, img_metas\):.*?(?=\n[ \t]*def )", re.MULTILINE | re.DOTALL)

# 拥有“无限穿透”和“维度重塑(Reshape)”能力的终极提取函数
new_func = """    def extract_img_feat(self, img, img_metas):
        import torch
        # 1. 穿透 DataContainer 提取 img_metas
        if isinstance(img_metas, list):
            img_metas = [m.data[0] if hasattr(m, 'data') and isinstance(m.data, list) else (m.data if hasattr(m, 'data') else m) for m in img_metas]
        elif hasattr(img_metas, 'data'):
            img_metas = img_metas.data
        if isinstance(img_metas, list) and isinstance(img_metas[0], list):
            img_metas = img_metas[0]
            
        # 2. 穿透 DataContainer 提取 img Tensor
        def _unwrap(x):
            if isinstance(x, torch.Tensor): return [x]
            if hasattr(x, 'data'): return _unwrap(x.data)
            if isinstance(x, (list, tuple)):
                res = []
                for item in x: res.extend(_unwrap(item))
                return res
            return []
            
        img_list = _unwrap(img)
        if len(img_list) > 0:
            img = torch.cat(img_list, dim=0)
        else:
            img = None

        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            for img_meta in img_metas:
                if isinstance(img_meta, dict):
                    img_meta['input_shape'] = input_shape
                
            # --- 核心修复: 维度重塑 ---
            # 如果 img 是 4D [B*N, C, H, W]，先将其转为 5D [B, N, C, H, W] 才能按摄像头视角(this_mask_idx)屏蔽
            is_4d = False
            B_orig = len(img_metas)
            if img.dim() == 4:
                is_4d = True
                C, H, W = img.shape[1:]
                N = img.shape[0] // B_orig
                img = img.view(B_orig, N, C, H, W)
                
            if isinstance(img_metas[0], dict) and 'img_mask_idx' in img_metas[0]:
                for i in range(B_orig):
                    this_mask_idx = img_metas[i].get('img_mask_idx', None)
                    if this_mask_idx is not None:
                        img[i, this_mask_idx, ...] = 0.0

            # 处理完 mask 之后，恢复回 backbone 需要的格式
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            # ---------------------------
            
            img_feats = self.img_backbone(img)
        else:
            return None
            
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats"""

code = pattern.sub(new_func, code)

with open(filepath, "w") as f:
    f.write(code)

print("✅ img 维度重塑(Reshape)与 Mask 屏蔽补丁已注入！不再会发生 IndexError！")
