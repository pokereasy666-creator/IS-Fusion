filepath = "mmdet3d/models/dense_heads/transfusion_head_v2.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# 往我们上一轮捏造的 mcb (mmdet.core.bbox) 模块中强行塞入 Assigner 的相关类
patch = """mcb.builder = mcbb
    
    # 骗过 Assigner 相关的导入
    try:
        from mmdet.models.task_modules.assigners import AssignResult, BaseAssigner, MaxIoUAssigner
        mcb.AssignResult = AssignResult
        mcb.BaseAssigner = BaseAssigner
        mcb.MaxIoUAssigner = MaxIoUAssigner
    except ImportError:
        pass
        
    try:
        from mmdet.models.task_modules.samplers import PseudoSampler
        mcb.PseudoSampler = PseudoSampler
    except ImportError:
        pass
"""

# 安全替换
if "mcb.AssignResult = AssignResult" not in code:
    code = code.replace("mcb.builder = mcbb", patch)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    print("✅ 破案了！已成功将 AssignResult 等核心组件放入伪装模块中！")
else:
    print("⚠️ 已经注入过了。")
