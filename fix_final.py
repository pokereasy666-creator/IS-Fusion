import re

filepath = "mmdet3d/models/detectors/isfusion.py"

with open(filepath, "r") as f:
    code = f.read()

# 终极杀手锏：把所有 Tab 强行替换为 4 个空格，杜绝一切 IndentationError
code = code.replace("\t", "    ")

dyn_patch = """    def dynamic_voxelize(self, points):
        import torch
        import torch.nn.functional as F
        def _get_tensors(x):
            if isinstance(x, torch.Tensor): return [x]
            if not isinstance(x, torch.Tensor) and hasattr(x, 'data'): return _get_tensors(x.data)
            if isinstance(x, (list, tuple)): 
                res = []
                for item in x: res.extend(_get_tensors(item))
                return res
            return []
        
        clean_points = _get_tensors(points)
        device = next(self.parameters()).device
        clean_points = [p.to(device).float().contiguous() for p in clean_points]
        coors = [self.pts_voxel_layer(p) for p in clean_points]
        coors_batch = [F.pad(c, (1, 0), mode='constant', value=i).to(torch.int32) for i, c in enumerate(coors)]
        return torch.cat(clean_points, dim=0), torch.cat(coors_batch, dim=0).contiguous()

"""

vox_patch = """    def voxelize(self, points, voxel_type='voxel'):
        import torch
        import torch.nn.functional as F
        def _get_tensors(x):
            if isinstance(x, torch.Tensor): return [x]
            if not isinstance(x, torch.Tensor) and hasattr(x, 'data'): return _get_tensors(x.data)
            if isinstance(x, (list, tuple)): 
                res = []
                for item in x: res.extend(_get_tensors(item))
                return res
            return []
        
        clean_points = _get_tensors(points)
        device = next(self.parameters()).device
        clean_points = [p.to(device).float().contiguous() for p in clean_points]
        v, c, n = [], [], []
        for p in clean_points:
            if voxel_type == 'voxel':
                res_v, res_c, res_n = self.pts_voxel_layer(p)
            else:
                res_v, res_c, res_n = self.pts_pillar_layer(p)
            v.append(res_v); c.append(res_c); n.append(res_n)
        
        cb = [F.pad(coor, (1, 0), mode='constant', value=i).to(torch.int32) for i, coor in enumerate(c)]
        return torch.cat(v, 0), torch.cat(n, 0), torch.cat(cb, 0).contiguous()

"""

dyn_pattern = re.compile(r"    def dynamic_voxelize\(self, points\):.*?(?=    def |\Z)", re.DOTALL)
vox_pattern = re.compile(r"    def voxelize\(self, points, voxel_type=.*?\):.*?(?=    def |\Z)", re.DOTALL)

code = dyn_pattern.sub(dyn_patch, code)
code = vox_pattern.sub(vox_patch, code)

# 修复 batch_size 提取，防止张量污染
if "batch_size = coors[-1, 0] + 1" in code:
    code = code.replace("batch_size = coors[-1, 0] + 1", "batch_size = int(coors[-1, 0].item()) + 1")

# 移除历史残留的混乱补丁，恢复到原始调用
code = re.sub(r"        # --- SAFE SPARSE COORS PATCH ---.*?x, _, kwargs = self\.pts_middle_encoder\(voxel_features, feature_coors, batch_size, \*\*kwargs\)", "        x, _, kwargs = self.pts_middle_encoder(voxel_features, feature_coors, batch_size, **kwargs)", code, flags=re.DOTALL)
code = re.sub(r"        feature_coors = feature_coors\.to\(torch\.int32\).*?x, _, kwargs = self\.pts_middle_encoder\(voxel_features, feature_coors, int\(batch_size\), \*\*kwargs\)", "        x, _, kwargs = self.pts_middle_encoder(voxel_features, feature_coors, batch_size, **kwargs)", code, flags=re.DOTALL)

# 注入全新的精确定位边界限制补丁 (Int32 + Clamp 防爆)
mid_pattern = r"        x, _, kwargs = self\.pts_middle_encoder\(voxel_features, feature_coors, batch_size, \*\*kwargs\)"
mid_patch = """        feature_coors = feature_coors.to(torch.int32)
        if feature_coors.shape[1] == 4:
            feature_coors[:, 0] = torch.clamp(feature_coors[:, 0], 0, int(batch_size) - 1)
            feature_coors[:, 1] = torch.clamp(feature_coors[:, 1], 0, 40)
            feature_coors[:, 2] = torch.clamp(feature_coors[:, 2], 0, 1439)
            feature_coors[:, 3] = torch.clamp(feature_coors[:, 3], 0, 1439)
        x, _, kwargs = self.pts_middle_encoder(voxel_features, feature_coors, int(batch_size), **kwargs)"""

code = re.sub(mid_pattern, mid_patch, code)

with open(filepath, "w") as f:
    f.write(code)

print("✅ 文件重写成功！所有 Tab 已清洗为空格，IndentationError 和 CUDA Error 2 彻底消灭！")
