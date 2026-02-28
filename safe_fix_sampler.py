import os, re, urllib.request

filepath = "mmdet3d/models/dense_heads/transfusion_head_v2.py"

# 1. 使用正确的官方仓库地址拉取文件
url = "https://raw.githubusercontent.com/yinjunbo/IS-Fusion/main/mmdet3d/models/dense_heads/transfusion_head_v2.py"
print(f"正在从 {url} 拉取原文件...")
try:
    urllib.request.urlretrieve(url, filepath)
    print("✅ 原始文件恢复成功！(Loss函数已找回)")
except Exception as e:
    print(f"⚠️ 下载失败 ({e})，尝试使用 git checkout 恢复...")
    if os.system(f"git checkout {filepath}") != 0:
        print("❌ 恢复失败，请手动将原压缩包里的 transfusion_head_v2.py 覆盖进来！")
        exit(1)

# 2. 读取完好无损的文件
with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# 3. 构造 MMDetection 3.x 兼容的 SamplingResult（跳过报错的 bbox_sampler）
new_code = """# --- MMDetection 3.x Compatibility Patch ---
        from mmdet.models.task_modules.samplers.sampling_result import SamplingResult
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        gt_flags = bboxes.new_zeros((bboxes.shape[0],), dtype=torch.uint8)
        sampling_result = SamplingResult(pos_inds=pos_inds, neg_inds=neg_inds, bboxes=bboxes, gt_bboxes=gt_bboxes_3d, assign_result=assign_result, gt_flags=gt_flags)
        # -------------------------------------------"""

# 4. 精确替换：只针对 sampling_result = self.bbox_sampler.sample(...) 这一块
content = re.sub(
    r'sampling_result\s*=\s*self\.bbox_sampler\.sample\([\s\S]*?type=[\'"]3D[\'"]\)', 
    new_code, 
    content
)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Sampler 接口已完美修复，准备起飞！")
