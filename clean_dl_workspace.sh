#!/bin/bash

echo "🚀 开始深度清理深度学习工作环境..."

# 1. 清理 Conda 所有的冗余缓存 (包、索引、压缩包)
echo "--- 正在清理 Conda 缓存 (pkgs & tarballs) ---"
conda clean --all -y

# 2. 清理 Pip 的全局缓存 (深度学习依赖包通常很大)
echo "--- 正在清理 Pip 缓存 ---"
pip cache purge

# 3. 清理当前目录及子目录下所有的 Python 编译缓存
echo "--- 正在清理 Python 编译缓存 (*.pyc, __pycache__) ---"
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# 4. (可选) 清理常见的编译中间目录，如 build/ 或 dist/
# 如果你经常编译 pcdet 或其他 C++ 扩展，取消下面行的注释
# echo "--- 正在清理 build 文件夹 ---"
# find . -type d -name "build" -exec rm -rf {} +

# 5. 清理系统缩略图缓存 (Ubuntu 长期运行会产生大量碎文件)
echo "--- 正在清理系统缩略图缓存 ---"
rm -rf ~/.cache/thumbnails/*

echo "✅ 清理完成！"
echo "当前磁盘剩余空间："
df -h / | grep /
