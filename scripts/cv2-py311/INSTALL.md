# OpenCV cv2：Python 3.11 Linux x86_64 完整离线安装包

严格按要求提供 `opencv-python-headless==4.11.0.86` 与 `numpy==1.26.4`，包含全部 Python wheel 依赖。目标为 CPython3.11 / Linux x86_64 / glibc2.17+；不适用于之前 Python3.10 容器、ARM64 或 Windows。OpenCV wheel 可能标记 cp37-abi3，这是官方稳定 ABI wheel，可被 CPython3.11 使用。

下载本 Release 的 ZIP 和 `.sha256`，复制到目标内网容器或服务器，在实际使用的 Python3.11 环境中执行：

```bash
sha256sum -c cv2_4.11.0.86_numpy1.26.4_py311_linux_x86_64_offline.zip.sha256
unzip cv2_4.11.0.86_numpy1.26.4_py311_linux_x86_64_offline.zip -d cv2_offline
cd cv2_offline
bash install.sh
```

指定解释器：`PYTHON_BIN=/path/to/python3.11 bash install.sh`。

手工安装也可：

```bash
python -m pip install --no-index --find-links wheels -r requirements.txt
python check_cv2.py
```

安装脚本先检查 Python、架构、校验和与 OpenCV 发行版冲突；若已有 opencv-python 或 opencv-contrib-* 则停止，不自动卸载。确认需要 headless 变体后再处理冲突，不能将多个发行版混装到同一个 cv2 命名空间。安装会将 OpenCV headless/NumPy 调整到上述指定版本，不安装或升级 PyTorch/CUDA。需要保留其他 NumPy 版本的环境应先评估兼容性。

headless 适合服务器、Docker 和离线图像处理，不提供 imshow 等 GUI 窗口功能。标准 manylinux 环境的 wheel 已带主要 OpenCV 二进制依赖；极简系统如缺 glibc/libstdc++ 等系统库，不能仅靠 pip 包修复。

发布前在干净 Python3.11 环境中，通过仅使用本地 wheels 的安装、pip check，以及实际 cv2 导入、PNG 编解码、灰度转换、缩放、旋转框几何、NumPy FFT 测试。验证日志随包提供；尚未在你的内网现有依赖环境实测。
