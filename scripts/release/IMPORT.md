# SAR Swin-T ARM64 A100 完整离线镜像

本发布包含可供 Docker 20.10 导入的完整镜像分卷，含 NVIDIA NGC PyTorch 24.10 ARM64/SBSA、CUDA 12.6、SAR Swin-T P1–P5 Faster R-CNN 源码及官方 Swin-T 预训练权重。镜像标签：`sar-swin-p1:arm64-cu126`。

## 下载和导入

下载本 Release 的全部附件，将它们一起复制到内网同一目录。分卷每份 1.5GB，必须全部下载，保持文件名不变。预留足够空间容纳分卷、合并压缩包和解压后的 Docker 镜像层。

```bash
bash import-swin.sh
```

脚本校验各附件 SHA256、按顺序合并分卷、校验完整压缩包，随后 `docker load` 并检查架构。也可手工执行：

```bash
sha256sum -c SHA256SUMS
cat SAR_SwinT_ARM64_A100_CUDA126_Image.tar.gz.part* > SAR_SwinT_ARM64_A100_CUDA126_Image.tar.gz
sha256sum -c SAR_SwinT_ARM64_A100_CUDA126_Image.tar.gz.sha256
docker load -i SAR_SwinT_ARM64_A100_CUDA126_Image.tar.gz
```

## 内网 GPU 验证

目标环境为麒麟 V10、鲲鹏920 ARM64、双 A100 40GB、驱动 560.35.03、Docker 20.10.1。宿主机须已配置适配 ARM64 的 NVIDIA Container Toolkit。

```bash
docker run --rm --gpus all --network none --shm-size=8g \
  sar-swin-p1:arm64-cu126 python check_env.py --require-cuda
tar -xzf SAR_SwinT_Import_Toolkit.tar.gz
```

后续使用工具包中的 `scripts/04_内网准备数据.sh`、`05_内网双卡训练.sh` 和 `06_内网评测.sh`，数据集由内网挂载。镜像包含 Swin-T 通用预训练权重，不包含已训练的 SAR 检测权重或数据集。

## 验证范围与溯源

构建在原生 ARM64 GitHub runner 完成，发布前断网检查依赖导入、CPU NMS/ROIAlign、官方权重严格加载及检测网络一次 CPU 前向。工程输入不代表真实 SAR 效果。详见日志、`BUILD-INFO.txt`、`image-inspect.json` 和 `base-digest.json`。

GitHub runner 没有 A100，CUDA NMS/ROIAlign、双卡训练和目标内核兼容性须按上述命令在内部服务器验证，未声称已通过 GPU 实测。
