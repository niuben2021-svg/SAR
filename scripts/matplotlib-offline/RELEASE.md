Matplotlib 3.8.4 完整离线安装包，适用于 CPython 3.10 / Linux x86_64 / glibc 2.17+。含全部 11 个 wheel，依赖按用户 constraints.txt 固定，包括 NumPy 1.26.4、Pillow 10.4.0。

下载 tar.gz 和 SHA256 文件，拷入目标 Docker，执行：
```bash
sha256sum -c matplotlib_py310_linux_x86_64_offline.tar.gz.sha256
tar -xzf matplotlib_py310_linux_x86_64_offline.tar.gz
cd matplotlib_py310_linux_x86_64_offline
bash install.sh
```

已在全新 CPython 3.10.22 x86_64 虚拟环境、沙箱隔离网络条件下，使用本地 wheel 安装、Agg 后端绘图并通过 pip check。实际内部 Docker 的其他依赖兼容性需在目标环境核对。本包不安装或升级 PyTorch/CUDA；固定 NumPy/Pillow 约束可能调整现有对应包版本。
