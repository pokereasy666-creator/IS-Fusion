import os
import re

# 定义需要修复的目录
target_dir = 'mmdet3d/datasets/'

# 定义替换规则库 (旧路径 -> 新路径的逻辑)
replacements = {
    # 1. 注册表搬家 (DATASETS)
    r'from mmdet.datasets import DATASETS': 
        'try:\n    from mmdet.datasets import DATASETS\nexcept ImportError:\n    from mmdet.registry import DATASETS',
    
    r'from mmseg.datasets import DATASETS as SEG_DATASETS':
        'try:\n    from mmseg.datasets import DATASETS as SEG_DATASETS\nexcept ImportError:\n    from mmseg.registry import DATASETS as SEG_DATASETS',

    # 2. 核心模块重组 (mmdet.core)
    r'from mmdet.core import eval_map':
        'try:\n    from mmdet.core import eval_map\nexcept ImportError:\n    from mmdet.evaluation import eval_map',
    
    r'from mmdet.core import (.*)':
        'try:\n    from mmdet.core import \\1\nexcept ImportError:\n    from mmdet.utils import \\1',

    # 3. 基础变换路径 (pipelines -> transforms)
    r'from mmdet.datasets.pipelines import (.*)':
        'try:\n    from mmdet.datasets.pipelines import \\1\nexcept ImportError:\n    from mmdet.datasets.transforms import \\1',
}

def fix_files():
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                new_content = content
                for pattern, subst in replacements.items():
                    # 避免重复修改：如果已经包含 try: ... except 就不再修改
                    if 'try:' in content and ('registry' in content or 'evaluation' in content):
                        continue
                    new_content = re.sub(pattern, subst, new_content)
                
                if new_content != content:
                    print(f"正在修复: {file_path}")
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)

if __name__ == '__main__':
    fix_files()
    print("\n✅ 批量清理完成！请重新运行 create_data 指令。")
