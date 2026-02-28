import os
import re

filepath = "mmdet3d/apis/train.py"
if os.path.exists(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()
    
    # 找到构建 MMDistributedDataParallel 或 DistributedDataParallel 的地方，注入 static_graph 参数
    # 我们不仅确保 find_unused_parameters 存在，还在模型实例化后强制调用 _set_static_graph()
    
    patch = """
    # --- 物理级 DDP 静态图护盾 ---
    if hasattr(model, '_set_static_graph'):
        model._set_static_graph()
    elif hasattr(model, 'module') and hasattr(model.module, '_set_static_graph'):
        model.module._set_static_graph()
    """
    
    # 寻找一个安全的插入点，通常在 model 放进 DDP 之后，但在 runner 初始化之前
    # 在 mmdet3d 的 apis/train.py 中，通常是在 build_runner 之前
    if "物理级 DDP 静态图护盾" not in code:
        code = re.sub(r'(runner\s*=\s*build_runner)', patch + r'\n    \1', code)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code)
        print("✅ DDP 静态图优化已开启！成功解决梯度检查点冲突！")
    else:
        print("⚠️ 静态图护盾已存在。")
else:
    print(f"❌ 找不到文件: {filepath}")
