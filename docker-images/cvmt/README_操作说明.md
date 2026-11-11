# CVMT on mmrota-swin 增量部署包（RTX 4090 × 4）

## 重要说明
本 ZIP 是**离线构建脚本+版本锁定配置**，并不含真实 PyPI/OpenMMLab 的二进制 whl，也不含作者 CVMT 源码、checkpoint 或原始 Docker 镜像，因为当前制作环境无法访问外部包站点。下载脚本须在**具有网络和 Docker 的 Linux x86_64 机器**上运行。运行之后会生成 `wheelhouse/`、`CVMT/` 和新镜像，才能离线部署。不要在已经运行 SSFG 的容器直接 pip uninstall。

底座假设为 `mmrota-swin:latest`，Python 3.10、PyTorch 2.0.1+CUDA11.8（已安装 GPU torch），适用于 RTX4090。新镜像使用 mmcv==2.0.1 (CUDA ops), mmengine==0.10.5, 作者自带 MMDetection 3.x，timm==0.9.16。依赖 wheel 与 CUDA 扩展要匹配 Python cp310 / Linux x86_64 / Torch2.0 / CUDA11.8。

## 方式A：外网服务器有 Docker 和底座镜像（**最推荐**）

1. 外网服务器先导入旧镜像（如果已有跳过）：

    docker load -i mmrota_swin.tar
    docker images

2. 解压本包，进入 `cvmt_incremental_env_4090`，确保 `BASE_IMAGE` 与 `docker images` 中实际标签一致：

    BASE_IMAGE=mmrota-swin:latest bash download_and_build.sh

    第一步会下载 CVMT 官方代码及 Python wheel；必须下载到带 CUDA ops 的 MMCV 2.0.1 **wheel**。找不到兼容的轮子会停止，避免悄悄装 CPU-only / 源码版本。整个 wheelhouse 可能较大（包含 torch 依赖链）。

3. 在外网机导出已构建镜像：

    docker save -o cvmt_cu118.tar cvmt:cu118

4. 传送 `cvmt_cu118.tar` 到你的内网机器，执行：

    docker load -i cvmt_cu118.tar
    bash run_container.sh

    镜像已经包含 CVMT 官方源码，位置 `/opt/CVMT`；宿主机 `/root/data1` 挂载到容器同路径。

5. 将之前 `cvmt_faircsar_A_patch_v2.zip` 解压后，运行转换/补丁/测试：

    python /root/data1/cvmt_faircsar_A_v2/prepare_faircsar_cvmt.py --source /root/data1/data --output /root/data1/data_cvmt --split both --workers 12
    python /root/data1/cvmt_faircsar_A_v2/patch_cvmt.py --repo /opt/CVMT --data-root /root/data1/data_cvmt/ --workers 12
    REPO=/opt/CVMT WEIGHTS=/root/data1/cvmt_faircsar.pth GPUS=1 WORKERS=12 bash /root/data1/cvmt_faircsar_A_v2/run_cvmt_test.sh

    `cvmt_faircsar.pth` 是示例路径，改为 FAIR-CSAR 对应的官方权重文件。单卡成功后 GPUS=4。

## 方式B：只转运离线源码和 wheels，不转 Docker 镜像

外网机完成 `download_and_build.sh` 后，把 `CVMT/`、`wheelhouse/`、`Dockerfile`、`verify_cvmt.py` 和 `offline_build.sh` 都传入内网；内网必须预先存在相同的 `mmrota-swin` 基础镜像；运行 `BASE_IMAGE=mmrota-swin:latest bash offline_build.sh`。但推荐方式A，避免内网编译和离线依赖错误。

## 常见错误

- 没有 mmcv-2.0.1 cp310 manylinux wheel：停止构建，需在外网机准备匹配 CUDA11.8/Torch2.0 的轮子或在带 `nvcc` 的 CUDA development 基础环境中编译；不能装 `mmcv-lite`。
- `No module named mmcv._ext`：误装不带 CUDA ops 版本。
- `mmdet.__file__` 显示 `/workspace/...`：有旧 PYTHONPATH 污染。使用 `/opt/CVMT` 优先。
- 4GPU worker CUDA 错误：应用此前 FAIR-CSAR v2 中 `patch_cvmt.py`，移除 `PackDetInputs_Dual` 在 worker 内 `.to(cuda:0)` 的行为。
- 论文 AP50：依赖正确对应的 FAIR-CSAR 作者权重、幅度图、相位映射和相同评测划分，环境成功不能保证精度复现。

## 来源
官方源码：https://github.com/RSIP-NJUPT/CVMT
MMDetection 3.x：https://github.com/open-mmlab/mmdetection
