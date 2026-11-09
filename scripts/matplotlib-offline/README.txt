Matplotlib 3.8.4 完整离线安装包
适用：CPython 3.10、Linux x86_64、glibc 2.17+（不适用 ARM64 或 Python 3.11/3.12）。
本包已经包含 matplotlib 及全部依赖，共 11 个 wheel，无需再次下载或编译。

在目标 Docker 容器内执行：
  tar -xzf matplotlib_py310_linux_x86_64_offline.tar.gz
  cd matplotlib_py310_linux_x86_64_offline
  bash install.sh

如需指定 Python：
  PYTHON_FOR_INSTALL=/path/to/python bash install.sh

安装器会检查 Python/架构、校验 wheel、仅从本地 wheels 目录安装，
然后使用 Agg 后端生成 matplotlib_offline_test.png。
绘图脚本可用：MPLBACKEND=Agg python plot_target.py --run-dir runs/target_v2_smoke

固定依赖见 constraints.txt；NumPy 固定 1.26.4，Pillow 固定 10.4.0。
不会安装或升级 PyTorch/CUDA；现有 Python 包若不符合约束，pip 可能调整对应版本。

验证：在全新 CPython 3.10.22 Linux x86_64 虚拟环境、网络被沙箱隔离时，
使用本包 install.sh 安装成功，成功输出 PNG，pip check 通过。
尚未在用户内部 Docker 的已有依赖环境中运行。
validation/ 中为真实验证日志和测试图。
