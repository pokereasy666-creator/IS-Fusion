import os

filepath = "mmdet3d/apis/train.py"
if os.path.exists(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()
    
    # 因为有了 Dummy Loss 保底，我们不再需要 DDP 去寻找未使用参数了
    # 关闭它，彻底解除和 Swin Transformer 梯度检查点的冲突！
    code = code.replace("find_unused_parameters=True)", "find_unused_parameters=False)")
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    print("✅ find_unused_parameters 已设置为 False，梯度检查点 (Checkpoint) 冲突已解除！")
else:
    print("❌ 找不到文件:", filepath)
