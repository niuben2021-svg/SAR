# SAR Swin-T RTX4090 CUDA 12.1 完整离线镜像

镜像标签：`sar-swin-p1:rtx4090-cu121`。本发布提供 `docker save` 导出的完整 AMD64 镜像分卷，包含 PyTorch 2.3.1、CUDA 12.1、cuDNN 8、TorchVision 0.18.1+cu121、NumPy 1.26.4、Pillow 10.4.0、指定制备包的 SAR 训练/推理代码及官方 Swin-T ImageNet 预训练权重。

目标服务器为 x86_64 / CentOS 7 / 4×RTX4090，驱动 535.146.02。宿主须配置 NVIDIA Container Toolkit。容器不包含宿主驱动、数据集或已经训练的 SAR 检测权重。

## 下载与内网导入

下载本 Release 的全部附件，保持文件名不变，复制到内网同一目录。每个镜像分卷最大 1.5GB，必须全部下载；不要使用 GitHub 自动生成的 Source code ZIP 代替镜像。为分卷、合并压缩包和 Docker 展开的镜像预留几十 GB 空间。

```bash
bash import-rtx4090.sh
```

该脚本检查所有附件的 SHA256、合并分卷、检查完整压缩包 SHA256、执行 `docker load` 并确认 amd64 架构。也可手工操作：

```bash
sha256sum -c SHA256SUMS
cat SAR_SwinT_RTX4090_CUDA121_Image.tar.gz.part* > SAR_SwinT_RTX4090_CUDA121_Image.tar.gz
sha256sum -c SAR_SwinT_RTX4090_CUDA121_Image.tar.gz.sha256
docker load -i SAR_SwinT_RTX4090_CUDA121_Image.tar.gz
```

## 内网 GPU 检查及训练入口

```bash
docker run --rm --gpus all --network none --shm-size=8g \
  sar-swin-p1:rtx4090-cu121 python check_env.py --require-gpus 4 --run-model
tar -xzf SAR_SwinT_RTX4090_Import_Toolkit.tar.gz
bash scripts/04_prepare_dataset.sh /data/dataset /data/sar_output
bash scripts/07_real_sample_smoketest.sh /data/dataset
bash scripts/05_train_4x4090.sh /data/dataset /data/sar_output 1
```

先通过 GPU 检查，再开始真实样本检查和训练。详见工具包 README。每卡 batch 从 1 起步，数据与输出通过宿主目录挂载。

## 已验证范围

GitHub x86_64 CPU runner 构建镜像，并在 `--network none` 下执行 pip 依赖检查、CPU NMS/ROIAlign、权重严格加载、Swin-T+P1 检测网络前向/反向/推理。导出后删除本次构建镜像和缓存，重新 `docker load` 并检查断网导入。CPU 工程测试使用合成输入，不代表 SAR 检测精度。

RTX4090 GPU、CUDA 算子、四卡训练、GPU 显存峰值与目标 CentOS 内核兼容性均未在 runner 实测，须在内部服务器验收。日志、实际依赖清单、基础镜像 digest、源代码 commit 和导出镜像信息均随附件提供。
