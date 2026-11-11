# CVMT 原镜像环境增量升级包（不是 Docker 镜像）

适用原始环境：Linux x86_64、Python3.10、PyTorch2.0.1+cu118、TorchVision0.15.2。只包含升级 wheel、官方 CVMT 源码和安装工具，不包含或重装 torch/torchvision/CUDA，不需要 docker load。

## 操作步骤
1. 下载 ZIP 和 .sha256，复制到内网宿主 /root/data1，核对：`sha256sum -c cvmt_upgrade_py310_cu118.zip.sha256`。
2. 建议保留原镜像，在原镜像上启动一个新容器，挂载安装包目录：
```bash
docker run --gpus all --shm-size=12g -it -v /root/data1:/root/data1 --name cvmt_upgrade mmrota-swin:latest bash
cd /root/data1
unzip cvmt_upgrade_py310_cu118.zip -d cvmt_upgrade
cd cvmt_upgrade
bash install.sh
export PYTHONPATH="$PWD/CVMT"
python verify_local.py
```
将 mmrota-swin:latest 替换为你的实际原始镜像标签。安装脚本会先检查原解释器/torch版本和文件校验，随后按附件要求卸载旧 mmrotate/mmdet/mmcv 系列，升级到 MMCV2.0.1、MMEngine0.10.5、timm0.9.16 和作者 MMDetection3.x。旧 MMRotate 0.x 与新 CVMT 不能在同一 Python 环境同时保留；旧镜像仍保留，勿对正在运行 SSFG 的容器直接执行。

## 保存升级结果
容器升级不会改变原镜像。不要用 --rm 启动需要保留的升级容器；退出后用 `docker start -ai cvmt_upgrade` 重新进入。也可在宿主执行 `docker commit cvmt_upgrade cvmt:cu118` 保存。安装源码位于挂载目录，PYTHONPATH 必须指向该目录；保存镜像不会打包宿主挂载数据，仍需保留并挂载该目录。

本包升级操作全部使用 --no-index。本包不含 FAIR-CSAR 数据、checkpoint 或附件提到但未提供的 patch_v2；它们需另行准备。镜像环境可用不代表完成端到端推理或精度复现。验证日志注明 CPU runner 无GPU，内网还须核对 CUDA NMS。官方源码固定 a75079317972ecffb86a0b2cedfe7258d2dce2e1，许可证随源码保留。
