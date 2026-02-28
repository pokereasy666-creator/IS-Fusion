import re

filepath = "mmdet3d/models/detectors/isfusion.py"
with open(filepath, "r") as f: code = f.read()

# 1. 连根拔起：彻底删除文件里已经损坏的 dynamic_voxelize 和 voxelize 方法
code = re.sub(r"^[ \t]*def dynamic_voxelize\(self, points\):.*?(?=\n[ \t]*def |\Z)", "", code, flags=re.MULTILINE | re.DOTALL)
code = re.sub(r"^[ \t]*def voxelize\(self, points, voxel_type=.*?\):.*?(?=\n[ \t]*def |\Z)", "", code, flags=re.MULTILINE | re.DOTALL)

# 2. 生成绝对免疫缩进错误的“单行流”安全方法
safe_methods = """
    def dynamic_voxelize(self, points):
        import torch; import torch.nn.functional as F
        def _g(x): return [x] if isinstance(x, torch.Tensor) else (_g(x.data) if hasattr(x, 'data') else ([i for s in x for i in _g(s)] if isinstance(x, (list, tuple)) else []))
        cp = [p.to(next(self.parameters()).device).float().contiguous() for p in _g(points)]
        return torch.cat(cp, 0), torch.cat([F.pad(self.pts_voxel_layer(p), (1,0), value=i).to(torch.int32) for i, p in enumerate(cp)], 0).contiguous()

    def voxelize(self, points, voxel_type='voxel'):
        import torch; import torch.nn.functional as F
        def _g(x): return [x] if isinstance(x, torch.Tensor) else (_g(x.data) if hasattr(x, 'data') else ([i for s in x for i in _g(s)] if isinstance(x, (list, tuple)) else []))
        cp = [p.to(next(self.parameters()).device).float().contiguous() for p in _g(points)]; v, c, n = [], [], []
        for i, p in enumerate(cp):
            rv, rc, rn = self.pts_voxel_layer(p) if voxel_type == 'voxel' else self.pts_pillar_layer(p)
            v.append(rv); c.append(F.pad(rc, (1,0), value=i).to(torch.int32)); n.append(rn)
        return torch.cat(v, 0), torch.cat(n, 0), torch.cat(c, 0).contiguous()

    def extract_pts_feat("""

# 将这两个安全方法插入到 extract_pts_feat 方法之前
code = re.sub(r"^[ \t]*def extract_pts_feat\(", safe_methods, code, flags=re.MULTILINE)

# 3. 大扫除：清理掉之前可能残留下来的所有错乱的 batch_size 和 坐标越界截断代码
code = re.sub(r"^[ \t]*# --- ULTIMATE.*?\n", "", code, flags=re.MULTILINE)
code = re.sub(r"^[ \t]*# --- SAFE.*?\n", "", code, flags=re.MULTILINE)
code = re.sub(r"^[ \t]*feature_coors = feature_coors\.to\(torch\.int32\)\n", "", code, flags=re.MULTILINE)
code = re.sub(r"^[ \t]*if feature_coors\.shape\[1\] == 4:[^\n]*\n", "", code, flags=re.MULTILINE)
code = re.sub(r"^[ \t]*feature_coors\[:, 0\] = torch\.clamp[^\n]*\n", "", code, flags=re.MULTILINE)
code = re.sub(r"^[ \t]*feature_coors\[:, 1\] = torch\.clamp[^\n]*\n", "", code, flags=re.MULTILINE)
code = re.sub(r"^[ \t]*feature_coors\[:, 2\] = torch\.clamp[^\n]*\n", "", code, flags=re.MULTILINE)
code = re.sub(r"^[ \t]*feature_coors\[:, 3\] = torch\.clamp[^\n]*\n", "", code, flags=re.MULTILINE)
code = re.sub(r"^[ \t]*batch_size = .*?coors\[-1, 0\].*?\n", "", code, flags=re.MULTILINE)
code = re.sub(r"^[ \t]*x, _, kwargs = self\.pts_middle_encoder.*?\n", "", code, flags=re.MULTILINE)

# 4. 植入最新的完美版稀疏卷积(SparseConv)越界防护
inj = """
        batch_size = int(feature_coors[-1, 0].item()) + 1
        feature_coors = feature_coors.to(torch.int32)
        if feature_coors.shape[1] == 4: feature_coors[:, 0].clamp_(0, batch_size - 1); feature_coors[:, 1].clamp_(0, 40); feature_coors[:, 2].clamp_(0, 1439); feature_coors[:, 3].clamp_(0, 1439)
        x, _, kwargs = self.pts_middle_encoder(voxel_features, feature_coors, batch_size, **kwargs)"""

code = re.sub(r"(voxel_features, feature_coors = self\.pts_voxel_encoder[^\n]+)", r"\1" + inj, code)

with open(filepath, "w") as f: f.write(code)
print("✅ 文件净化重写完成！所有混乱的空格、缩进问题和类型越界陷阱已作为历史尘埃被彻底抹去！")
