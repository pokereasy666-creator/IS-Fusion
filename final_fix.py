import os

filepath = "mmdet3d/models/detectors/isfusion.py"
with open(filepath, 'r') as f:
    lines = f.readlines()

new_lines = []
skip_until = None

for i, line in enumerate(lines):
    if skip_until:
        if skip_until in line:
            skip_until = None
        continue

    # 1. 修复 dynamic_voxelize (从定义开始到 return 结束)
    if "def dynamic_voxelize(self, points):" in line:
        new_lines.append("    def dynamic_voxelize(self, points):\n")
        new_lines.append("        import torch\n")
        new_lines.append("        import torch.nn.functional as F\n")
        new_lines.append("        def _get_tensors(x):\n")
        new_lines.append("            if isinstance(x, torch.Tensor): return [x]\n")
        new_lines.append("            if not isinstance(x, torch.Tensor) and hasattr(x, 'data'): return _get_tensors(x.data)\n")
        new_lines.append("            if isinstance(x, (list, tuple)): \n")
        new_lines.append("                res = []; [res.extend(_get_tensors(it)) for it in x]; return res\n")
        new_lines.append("            return []\n")
        new_lines.append("        clean_points = _get_tensors(points)\n")
        new_lines.append("        device = next(self.parameters()).device\n")
        new_lines.append("        clean_points = [p.to(device).float().contiguous() for p in clean_points]\n")
        new_lines.append("        coors = [self.pts_voxel_layer(p) for p in clean_points]\n")
        new_lines.append("        coors_batch = [F.pad(c, (1, 0), mode='constant', value=idx).to(torch.int32) for idx, c in enumerate(coors)]\n")
        new_lines.append("        return torch.cat(clean_points, dim=0), torch.cat(coors_batch, dim=0).contiguous()\n")
        skip_until = "return " # 跳过旧的函数体直到遇到 return
        continue

    # 2. 修复 voxelize (同上)
    if "def voxelize(self, points, voxel_type=" in line:
        new_lines.append("    def voxelize(self, points, voxel_type='voxel'):\n")
        new_lines.append("        import torch\n")
        new_lines.append("        import torch.nn.functional as F\n")
        new_lines.append("        def _get_tensors(x):\n")
        new_lines.append("            if isinstance(x, torch.Tensor): return [x]\n")
        new_lines.append("            if not isinstance(x, torch.Tensor) and hasattr(x, 'data'): return _get_tensors(x.data)\n")
        new_lines.append("            if isinstance(x, (list, tuple)): \n")
        new_lines.append("                res = []; [res.extend(_get_tensors(it)) for it in x]; return res\n")
        new_lines.append("            return []\n")
        new_lines.append("        clean_points = _get_tensors(points)\n")
        new_lines.append("        device = next(self.parameters()).device\n")
        new_lines.append("        clean_points = [p.to(device).float().contiguous() for p in clean_points]\n")
        new_lines.append("        v, c, n = [], [], []\n")
        new_lines.append("        for p in clean_points:\n")
        new_lines.append("            res_v, res_c, res_n = self.pts_voxel_layer(p) if voxel_type=='voxel' else self.pts_pillar_layer(p)\n")
        new_lines.append("            v.append(res_v); c.append(res_c); n.append(res_n)\n")
        new_lines.append("        cb = [F.pad(coor, (1, 0), mode='constant', value=idx).to(torch.int32) for idx, coor in enumerate(c)]\n")
        new_lines.append("        return torch.cat(v, 0), torch.cat(n, 0), torch.cat(cb, 0).contiguous()\n")
        skip_until = "return "
        continue

    # 3. 修复 pts_middle_encoder 越界
    if "self.pts_middle_encoder(voxel_features, feature_coors, batch_size" in line:
        # 先把坐标转为 int32 并限制范围 (41, 1440, 1440)
        new_lines.append("        feature_coors = feature_coors.to(torch.int32)\n")
        new_lines.append("        if feature_coors.shape[1] == 4:\n")
        new_lines.append("            feature_coors[:, 0] = torch.clamp(feature_coors[:, 0], 0, int(batch_size) - 1)\n")
        new_lines.append("            feature_coors[:, 1] = torch.clamp(feature_coors[:, 1], 0, 40)\n")
        new_lines.append("            feature_coors[:, 2] = torch.clamp(feature_coors[:, 2], 0, 1439)\n")
        new_lines.append("            feature_coors[:, 3] = torch.clamp(feature_coors[:, 3], 0, 1439)\n")
        new_lines.append("        x, _, kwargs = self.pts_middle_encoder(voxel_features, feature_coors, int(batch_size), **kwargs)\n")
        continue

    new_lines.append(line)

with open(filepath, 'w') as f:
    f.writelines(new_lines)

print("✅ 终极补丁：缩进已完全重置为标准4空格，且注入了 int32 和范围检查防护。")
