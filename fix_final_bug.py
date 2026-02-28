import os
import re

# 1. 撤销 second_fpn.py 中引发局部变量作用域冲突的 hack
fpn_file = "mmdet3d/models/necks/second_fpn.py"
if os.path.exists(fpn_file):
    with open(fpn_file, "r", encoding="utf-8") as f:
        code = f.read()
    
    # 删掉那个惹祸的 import torch 和强制垃圾回收
    bad_code = "del ups\n        import gc, torch\n        gc.collect()\n        torch.cuda.empty_cache()\n        out = out.permute(0, 1, 3, 2).contiguous()"
    good_code = "out = out.permute(0, 1, 3, 2).contiguous()"
    
    code = code.replace(bad_code, good_code)
    # 正则兜底
    code = re.sub(r'del ups\s+import gc, torch\s+gc\.collect\(\)\s+torch\.cuda\.empty_cache\(\)\s+out = out\.permute', 'out = out.permute', code)
    
    with open(fpn_file, "w", encoding="utf-8") as f:
        f.write(code)
    print("✅ 局部变量污染已清除！")

# 2. 在 train.py 运行时动态强改 cfg，无视配置文件继承，实现绝对截断！
train_file = "tools/train.py"
with open(train_file, "r", encoding="utf-8") as f:
    train_code = f.read()

voxel_hack = """
    # --- 运行时强行修改字典，筑起绝对的显存防波堤 ---
    def limit_voxels(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if k == 'max_voxels':
                    if isinstance(v, tuple) and len(v) == 2:
                        d[k] = (30000, 60000)
                    elif isinstance(v, list) and len(v) == 2:
                        d[k] = [30000, 60000]
                else:
                    limit_voxels(v)
        elif isinstance(d, list):
            for item in d:
                limit_voxels(item)
    limit_voxels(cfg._cfg_dict)
    print("🚀 [HOTFIX] 运行时强行将所有 max_voxels 压低到 30000/60000，绝对防御极限拥挤帧的 OOM！")
    # ---------------------------------------------------
"""

if "limit_voxels(cfg._cfg_dict)" not in train_code:
    train_code = train_code.replace("cfg = Config.fromfile(args.config)", "cfg = Config.fromfile(args.config)\n" + voxel_hack)
    with open(train_file, "w", encoding="utf-8") as f:
        f.write(train_code)
    print("✅ 运行时绝对防波堤已部署完毕！")
else:
    print("✅ 运行时防波堤已存在。")
