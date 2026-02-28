import re

filepath = "mmdet3d/models/detectors/isfusion.py"
with open(filepath, "r") as f:
    code = f.read()

# 使用正则匹配出整个坏掉的 extract_img_feat 函数（直到遇到下一个 def）
pattern = re.compile(r"^[ \t]*def extract_img_feat\(self, img, img_metas\):.*?(?=\n[ \t]*def )", re.MULTILINE | re.DOTALL)

# 完美对齐的、不含任何悬空 if 的全新函数
perfect_func = """    def extract_img_feat(self, img, img_metas):
        # 1. 彻底解包 img_metas
        if isinstance(img_metas, list) and hasattr(img_metas[0], 'data'):
            img_metas = [m.data[0] if isinstance(m.data, list) else m.data for m in img_metas]
        elif hasattr(img_metas, 'data'):
            img_metas = img_metas.data
        if isinstance(img_metas, list) and isinstance(img_metas[0], list):
            img_metas = img_metas[0]
            
        # 2. 彻底解包 img，让它变回纯净的 Tensor
        if hasattr(img, 'data'):
            img = img.data[0] if isinstance(img.data, list) else img.data
            
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            for img_meta in img_metas:
                if isinstance(img_meta, dict):
                    img_meta['input_shape'] = input_shape
                
            if isinstance(img_metas[0], dict) and 'img_mask_idx' in img_metas[0]:
                B = img.size(0)
                for i in range(B):
                    this_mask_idx = img_metas[i].get('img_mask_idx', None)
                    if this_mask_idx is not None:
                        img[i][this_mask_idx, ...] = 0.0

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img)
        else:
            return None
            
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats"""

# 执行整块替换
new_code = pattern.sub(perfect_func, code)

with open(filepath, "w") as f:
    f.write(new_code)

print("✅ 整个图像特征提取模块已重写！缩进错误彻底清除，且 DataContainer 已完美解包！")
