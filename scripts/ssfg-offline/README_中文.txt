SSFG / Swin-T 旋转框工程 Python 依赖：离线下载+安装整包
================================================

本工具准备的五类Python包：
  opencv-python-headless==4.10.0.84    -- cv2; 旋转框几何与Rayleigh量化
  numpy                                  -- 数值数组、FFT、PCA
  scipy                                  -- Taylor窗、MAT v5读取
  Pillow                                 -- PNG图像加载
  h5py                                   -- 可选MAT v7.3 HDF5读取

以及pip自动解析的附属Python依赖。如果内网已安装这些包，安装器不会覆盖。
现有PyTorch/TorchVision是CUDA配套版本，应保留，不包含在新包中。
请在当前训练用的同一个Python环境中执行，不要混用 conda base / venv / 系统Python。

步骤1. 【内网训练机】先执行
  python probe_target.py
  输出 target_env.json。把这个文件和本工具其余几个文件复制到有网络的Linux/Windows机器。

步骤2. 【外网电脑/服务器】安装pip后，在同一个工具目录执行
  python download_on_online.py target_env.json
  它会按内网Python版本、架构下载对应Linux wheel，包括所有相关Python依赖。
  完成后生成 ssfg_full_offline_wheels.zip (真正包含WHL二进制，大小可能100-200MB以上)。
  注意：只有该生成的zip才是真正全套可离线安装包，当前脚本zip不是。
  外网电脑无需装GPU，无需装内网对应的Python版本，pip使用cross-target下载。

步骤3. 【内网训练机】把 ssfg_full_offline_wheels.zip 传入内网，解压：
  unzip -o ssfg_full_offline_wheels.zip -d offline_deps
  cd offline_deps
  bash install_offline.sh
  或者：PYTHON_BIN=/opt/venv/bin/python bash install_offline.sh

步骤4. 【验证】
  python verify_train_env.py
  cd /root/data1/0swin/ssfg_swin_rotated
  python test_smoke.py

离线安装过程中使用 --no-index --no-deps，不会访问互联网，不会改变既有的CUDA和PyTorch。
`install_offline.py` 只安装缺失的Python包；如果已经安装但导入损坏，会停下而非盲目覆盖。
注意：PyTorch/TorchVision预先可用是前提。如果版本对不上，不要直接下载新的torch覆盖现有环境。

如果网络侧pip download报错：
  1. 检查内网 target_env.json 是否正确、目标是 Linux x86_64/aarch64。
  2. 检查网络侧pip版本：python -m pip --version，建议pip>=21。
  3. 检查你已有 NumPy/SciPy 版本是否存在匹配目标Python的预编译wheel。
  4. 目标Python 3.13及以上另行核对OpenCV兼容组合，不做盲目安装。

OS级动态库（libstdc++、libgcc_s、glibc等）不属于pip依赖，也不能被Python wheel完全保证。
OpenCV headless wheel通常已包含其主要二进制组件，但非标准精简容器如缺系统库仍需另行处理。
