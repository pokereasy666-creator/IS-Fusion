import os

fpn_file = "mmdet3d/models/necks/second_fpn.py"
if os.path.exists(fpn_file):
    with open(fpn_file, "r", encoding="utf-8") as f:
        code = f.read()
    
    # 精准定位到需要过河拆桥的地方
    old_str = "out = out.permute(0, 1, 3, 2).contiguous()"
    
    # 注入：删掉上游特征图，强制回收，再做 contiguous
    new_str = """del ups
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        out = out.permute(0, 1, 3, 2).contiguous()"""
    
    if "del ups" not in code:
        code = code.replace(old_str, new_str)
        with open(fpn_file, "w", encoding="utf-8") as f:
            f.write(code)
        print("✅ 过河拆桥战术部署成功！已提前释放中间拼接变量！")
    else:
        print("⚠️ 补丁已存在。")
