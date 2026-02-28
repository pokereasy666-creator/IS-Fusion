import urllib.request
import os

filepath = "mmdet3d/models/dense_heads/transfusion_head_v2.py"

# 1. 尝试从真实的 master 分支拉取原始代码
url_master = "https://raw.githubusercontent.com/yinjunbo/IS-Fusion/master/mmdet3d/models/dense_heads/transfusion_head_v2.py"
url_main = "https://raw.githubusercontent.com/yinjunbo/IS-Fusion/main/mmdet3d/models/dense_heads/transfusion_head_v2.py"

print("正在从 GitHub 恢复被误删的文件...")
try:
    urllib.request.urlretrieve(url_master, filepath)
except Exception:
    try:
        urllib.request.urlretrieve(url_main, filepath)
    except Exception as e:
        print("下载失败，请手动从原版压缩包里解压 transfusion_head_v2.py 覆盖回来。")
        exit(1)

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

if "def loss(" not in content:
    print("恢复的文件不完整。")
    exit(1)
print("✅ 原始文件恢复成功！被删掉的 loss 函数回来了。")

# 2. 智能括号匹配，精准替换
start_str = "sampling_result = self.bbox_sampler.sample("
while True:
    start_idx = content.find(start_str)
    if start_idx == -1:
        break
    
    paren_count = 1
    end_idx = -1
    search_start = start_idx + len(start_str)
    for i in range(search_start, len(content)):
        if content[i] == '(':
            paren_count += 1
        elif content[i] == ')':
            paren_count -= 1
            if paren_count == 0:
                end_idx = i
                break
                
    if end_idx != -1:
        # 获取当前行的前导缩进
        indent = ""
        idx = start_idx - 1
        while idx >= 0 and content[idx] != '\n':
            if content[idx] in " \t":
                indent = content[idx] + indent
            else:
                indent = ""
            idx -= 1
            
        new_code = """# --- MMDetection 3.x Compatibility Patch ---
from mmdet.models.task_modules.samplers.sampling_result import SamplingResult
pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
gt_flags = bboxes.new_zeros((bboxes.shape[0],), dtype=torch.uint8)
sampling_result = SamplingResult(pos_inds=pos_inds, neg_inds=neg_inds, bboxes=bboxes, gt_bboxes=gt_bboxes_3d, assign_result=assign_result, gt_flags=gt_flags)"""
        
        # 完美对齐缩进
        new_code = new_code.replace('\n', '\n' + indent)
        content = content[:start_idx] + new_code + content[end_idx+1:]
    else:
        break

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)
print("✅ Sampler 接口修复成功！这次不仅跳过了报错，还保全了所有其他代码！")
