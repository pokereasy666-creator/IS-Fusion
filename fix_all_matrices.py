import re

filepath = "mmdet3d/models/middle_encoders/fusion_encoder.py"
with open(filepath, "r") as f:
    code = f.read()

# 替换所有形如: cur_XXX = XXX[b] if not isinstance(XXX, list) else XXX[0][b]
# 统一替换为: cur_XXX = _get_item(XXX, b)

patterns_to_replace = [
    r"cur_img_aug_matrix\s*=\s*img_aug_matrix\[b\].*?img_aug_matrix\[0\]\[b\]",
    r"cur_lidar2img\s*=\s*lidar2img\[b\].*?lidar2img\[0\]\[b\]",
    r"cur_lidar_aug_matrix\s*=\s*lidar_aug_matrix\[b\].*?lidar_aug_matrix\[0\]\[b\]",
    r"cur_camera2ego\s*=\s*camera2ego\[b\].*?camera2ego\[0\]\[b\]",
    r"cur_lidar2ego\s*=\s*lidar2ego\[b\].*?lidar2ego\[0\]\[b\]",
    r"cur_lidar2camera\s*=\s*lidar2camera\[b\].*?lidar2camera\[0\]\[b\]"
]

for p in patterns_to_replace:
    # 提取 XXX 的名字
    matrix_name = p.split(r"\s*=\s*")[0].replace("cur_", "").replace(r"cur\_", "")
    replacement = f"cur_{matrix_name} = _get_item({matrix_name}, b)"
    code = re.sub(p, replacement, code, flags=re.MULTILINE)

# 为了防止 _get_val 还没有提取 lidar_aug_matrix 等变量，我们需要在 _get_val 定义后加上提取语句
extract_block_target = "image_size = _get_val('input_shape')"
if "lidar_aug_matrix = _get_val('lidar_aug_matrix')" not in code:
    extra_extracts = """
        lidar_aug_matrix = _get_val('lidar_aug_matrix')
        camera2ego = _get_val('camera2ego')
        lidar2ego = _get_val('lidar2ego')
        lidar2camera = _get_val('lidar2camera')
        
        if isinstance(lidar_aug_matrix, torch.Tensor): lidar_aug_matrix = [lidar_aug_matrix]
        if isinstance(camera2ego, torch.Tensor): camera2ego = [camera2ego]
        if isinstance(lidar2ego, torch.Tensor): lidar2ego = [lidar2ego]
        if isinstance(lidar2camera, torch.Tensor): lidar2camera = [lidar2camera]
"""
    code = code.replace(extract_block_target, extract_block_target + extra_extracts)

with open(filepath, "w") as f:
    f.write(code)

print("✅ fix_all_matrices.py 执行成功：所有矩阵变量已全部应用安全解包！")
