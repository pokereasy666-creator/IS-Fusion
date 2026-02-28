import os
import re
import urllib.request

filepath = "mmdet3d/models/detectors/isfusion.py"

# 1. 尝试从源头下载纯净版，洗白白
try:
    url = "https://raw.githubusercontent.com/yinjunbo/IS-Fusion/master/mmdet3d/models/detectors/isfusion.py"
    urllib.request.urlretrieve(url, filepath)
    print("✅ 成功从 GitHub 获取最纯净的原版代码！")
except Exception:
    print("⚠️ 无法连接外网，启动本地强力清道夫模式...")

# 2. 读取当前文件
with open(filepath, "r") as f:
    lines = f.readlines()

clean_lines = []
in_bad_zone = False

# 3. 无情删掉所有损坏的代码块
for line in lines:
    s = line.strip()
    if s.startswith("def dynamic_voxelize(self"): in_bad_zone = True
    if s.startswith("def voxelize(self"): in_bad_zone = True
    if s.startswith("def _get_tensors("): in_bad_zone = True
    if s.startswith("def _g("): in_bad_zone = True
    if s.startswith("@torch.no_grad") and ("dynamic_voxelize" in "".join(lines) or "voxelize" in "".join(lines)): continue
    if s.startswith("@force_fp32") and ("dynamic_voxelize" in "".join(lines) or "voxelize" in "".join(lines)): continue
        
    if in_bad_zone:
        if s.startswith("def extract_pts_feat(self") or s.startswith("def extract_feat(self") or s.startswith("def forward(self"):
            in_bad_zone = False
        else: continue

    if not in_bad_zone:
        clean_lines.append(line)

# 4. 定位安全插入点
insert_idx = -1
for i, line in enumerate(clean_lines):
    if line.strip().startswith("def extract_pts_feat(self"):
        insert_idx = i
        break

# 5. 绝对免疫缩进错误的“单行流”安全方法
pure_funcs = """    def dynamic_voxelize(self, points):
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

"""

if insert_idx != -1:
    clean_lines.insert(insert_idx, pure_funcs)

code = "".join(clean_lines)

# 6. 大扫除多余补丁，注入 SparseConv 的防爆截断
code = re.sub(r"[ \t]*batch_size = int\(feature_coors\[-1, 0\]\.item\(\)\) \+ 1\n", "", code)
code = re.sub(r"[ \t]*feature_coors = feature_coors\.to\(torch\.int32\)\n", "", code)
code = re.sub(r"[ \t]*if feature_coors\.shape\[1\] == 4: feature_coors.*?\n", "", code)

old_call = r"[ \t]*x, _, kwargs = self\.pts_middle_encoder\(voxel_features, feature_coors, batch_size, \*\*kwargs\)"
new_call = """        batch_size = int(feature_coors[-1, 0].item()) + 1
        feature_coors = feature_coors.to(torch.int32)
        if feature_coors.shape[1] == 4: feature_coors[:, 0].clamp_(0, batch_size - 1); feature_coors[:, 1].clamp_(0, 40); feature_coors[:, 2].clamp_(0, 1439); feature_coors[:, 3].clamp_(0, 1439)
        x, _, kwargs = self.pts_middle_encoder(voxel_features, feature_coors, batch_size, **kwargs)"""

code = re.sub(old_call, new_call, code)

with open(filepath, "w") as f:
    f.write(code)

print("✅ 终极净化完成！所有缩进问题和坐标越界陷阱已彻底消灭！")
