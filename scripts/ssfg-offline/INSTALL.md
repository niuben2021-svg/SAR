# SSFG / Swin-T 旋转框工程：完整离线 Python 依赖

适用 CPython 3.10 / Linux x86_64 / glibc 2.17+，按本会话此前 RTX4090 CUDA12.1 镜像准备。**未收到 target_env.json，本包 target_env.json 是声明的镜像基线，不是对内网机器的实测记录。** 如果训练环境已经换为 ARM64 或其他 Python 版本，请先运行 probe_target.py 并按实测结果另行制包。

包含全部 5 个 wheel：OpenCV headless 4.10.0.84、NumPy 1.26.4、SciPy 1.13.1、Pillow 10.4.0、h5py 3.11.0。这些固定版本的 Python 依赖仅需 NumPy，已一并提供。现有 CUDA 配套 torch/torchvision 由原训练容器提供，本包不会下载安装它们。

下载 ZIP 和 SHA256 文件并复制到内网，执行：

```bash
sha256sum -c ssfg_full_offline_wheels.zip.sha256
unzip ssfg_full_offline_wheels.zip -d offline_deps
cd offline_deps
bash install_offline.sh
python verify_train_env.py
```

需在原训练解释器中执行；也可 `PYTHON_BIN=/opt/venv/bin/python bash install_offline.sh`。

安装器保留附件中的 GPU 前置检查：先确认现有 PyTorch、TorchVision 和 CUDA 可用，再安装缺失包。可正常导入的现有包保留；存在其他 OpenCV 发行版时停止。安装仅使用本地 wheels，带 --no-index --no-deps，不升级 torch/CUDA。已有损坏安装需排查，不能将本包当作强制修复器。

发布前验证：在干净 CPython3.10 虚拟环境中，仅用本地 wheel 安装全部依赖，运行 pip check，实测 FFT/SVD、旋转框几何、Taylor 窗、MAT v5、HDF5 与 PNG 读写。附 offline-tests.log。未伪造 GPU 环境；完整 install_offline.py 的 GPU 前置检查及 verify_train_env.py 必须在内部服务器运行。代码 test_smoke.py 需要你的工程和数据，未在此代跑。
