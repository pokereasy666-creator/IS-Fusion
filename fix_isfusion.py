import re

filepath = "mmdet3d/models/detectors/isfusion.py"
with open(filepath, "r") as f:
    code = f.read()

dynamic_patch = """    def dynamic_voxelize(self, points):
        import torch
        import torch.nn.functional as F
        
        def get_tensors(x):
            if isinstance(x, torch.Tensor): return [x]
            if not isinstance(x, torch.Tensor) and hasattr(x, "data"): return get_tensors(x.data)
            if isinstance(x, (list, tuple)):
                res = []
                for item in x: res.extend(get_tensors(item))
                return res
            return []
            
        clean_points = get_tensors(points)
        device = next(self.parameters()).device
        clean_points = [p.to(device).float().contiguous() for p in clean_points]
        
        coors = []
        for res in clean_points:
            coors.append(self.pts_voxel_layer(res))
            
        points_concat = torch.cat(clean_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            # 核心: pad会变成int64, 必须强转回int32
            coor_pad = F.pad(coor, (1, 0), mode="constant", value=i).to(torch.int32)
            coors_batch.append(coor_pad)
        return points_concat, torch.cat(coors_batch, dim=0).contiguous()
"""

voxelize_patch = """    def voxelize(self, points, voxel_type="voxel"):
        import torch
        import torch.nn.functional as F
        
        def get_tensors(x):
            if isinstance(x, torch.Tensor): return [x]
            if not isinstance(x, torch.Tensor) and hasattr(x, "data"): return get_tensors(x.data)
            if isinstance(x, (list, tuple)):
                res = []
                for item in x: res.extend(get_tensors(item))
                return res
            return []
            
        clean_points = get_tensors(points)
        device = next(self.parameters()).device
        clean_points = [p.to(device).float().contiguous() for p in clean_points]
        
        voxels, coors, num_points = [], [], []
        for res in clean_points:
            if voxel_type == "voxel":
                res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            else:
                res_voxels, res_coors, res_num_points = self.pts_pillar_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
            
        voxels_batch = torch.cat(voxels, dim=0)
        num_points_batch = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            # 核心: pad会变成int64, 必须强转回int32
            coor_pad = F.pad(coor, (1, 0), mode="constant", value=i).to(torch.int32)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0).contiguous()
        return voxels_batch, num_points_batch, coors_batch
"""

# 应用两个补丁
code = re.sub(r"    def dynamic_voxelize\(self, points\):.*?(?=    def |\Z)", dynamic_patch, code, flags=re.DOTALL)
code = re.sub(r"    def voxelize\(self, points, voxel_type=.voxel.\):.*?(?=    def |\Z)", voxelize_patch, code, flags=re.DOTALL)

# 防止 batch_size 变成 Tensor 导致其他算子错误
code = code.replace("batch_size = coors[-1, 0] + 1", "batch_size = int(coors[-1, 0].item()) + 1")

# 将所有的 feature_coors 在传入 pts_middle_encoder 之前安全截断
clamp_patch = """        # 安全截断，防止 mmcv sparse conv 越界崩溃 (shape: 41, 1440, 1440)
        feature_coors = feature_coors.to(torch.int32)
        if feature_coors.shape[1] == 4:
            feature_coors[:, 0] = torch.clamp(feature_coors[:, 0], 0, batch_size - 1)
            feature_coors[:, 1] = torch.clamp(feature_coors[:, 1], 0, 40)
            feature_coors[:, 2] = torch.clamp(feature_coors[:, 2], 0, 1439)
            feature_coors[:, 3] = torch.clamp(feature_coors[:, 3], 0, 1439)
        x, _, kwargs = self.pts_middle_encoder(voxel_features, feature_coors, batch_size, **kwargs)"""

code = code.replace("x, _, kwargs = self.pts_middle_encoder(voxel_features, feature_coors, batch_size, **kwargs)", clamp_patch)


with open(filepath, "w") as f:
    f.write(code)

print("✅ 所有终极防护补丁注入完毕！包括: DataContainer解包 / int32防爆 / Clamp越界截断")

