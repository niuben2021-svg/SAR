# Qwen3-VL-4B Linux AMD64 镜像下载

`amd64` = `x86_64` = 64 位 x86，可用于 Intel 或 AMD x86_64 CPU。

已读取公开 Release 镜像归档内配置，确认 `os=linux`、`architecture=amd64`，证据见 architecture-verification.json。

镜像为原已发布版本，无需重新构建。总大小 4,606,524,301 字节，约 4.61 GB。GitHub 普通仓库不存放 GB 级镜像；本目录索引 Release 附件。

| 文件 | 大小（字节） |
| --- | ---: |
| [DOWNLOAD_README.md](https://github.com/niuben2021-svg/SAR/releases/download/qwen3vl4b-offline-20261023/DOWNLOAD_README.md) | 1803 |
| [qwen3vl4b-offline-guide.md](https://github.com/niuben2021-svg/SAR/releases/download/qwen3vl4b-offline-20261023/qwen3vl4b-offline-guide.md) | 14538 |
| [sar-vlm-qwen3vl4b-image.tar.gz.part01](https://github.com/niuben2021-svg/SAR/releases/download/qwen3vl4b-offline-20261023/sar-vlm-qwen3vl4b-image.tar.gz.part01) | 1500000000 |
| [sar-vlm-qwen3vl4b-image.tar.gz.part02](https://github.com/niuben2021-svg/SAR/releases/download/qwen3vl4b-offline-20261023/sar-vlm-qwen3vl4b-image.tar.gz.part02) | 1500000000 |
| [sar-vlm-qwen3vl4b-image.tar.gz.part03](https://github.com/niuben2021-svg/SAR/releases/download/qwen3vl4b-offline-20261023/sar-vlm-qwen3vl4b-image.tar.gz.part03) | 1500000000 |
| [sar-vlm-qwen3vl4b-image.tar.gz.part04](https://github.com/niuben2021-svg/SAR/releases/download/qwen3vl4b-offline-20261023/sar-vlm-qwen3vl4b-image.tar.gz.part04) | 106524301 |
| [sar-vlm-qwen3vl4b-image.tar.gz.sha256](https://github.com/niuben2021-svg/SAR/releases/download/qwen3vl4b-offline-20261023/sar-vlm-qwen3vl4b-image.tar.gz.sha256) | 97 |
| [sar-vlm-qwen3vl4b-source.tar.gz](https://github.com/niuben2021-svg/SAR/releases/download/qwen3vl4b-offline-20261023/sar-vlm-qwen3vl4b-source.tar.gz) | 98950 |
| [sar-vlm-qwen3vl4b-source.tar.gz.sha256](https://github.com/niuben2021-svg/SAR/releases/download/qwen3vl4b-offline-20261023/sar-vlm-qwen3vl4b-source.tar.gz.sha256) | 98 |
| [SHA256SUMS](https://github.com/niuben2021-svg/SAR/releases/download/qwen3vl4b-offline-20261023/SHA256SUMS) | 901 |

## 下载和导入

下载本目录的 download.sh、import.sh 到同一目录，然后在外网 Linux 主机执行：

```bash
bash download.sh ./sar-amd64-delivery
```

将下载目录和 import.sh 拷到内网 Linux x86_64 主机，执行：

```bash
bash import.sh ./sar-amd64-delivery
docker image inspect sar-vlm:qwen3vl4b --format '{{.Os}}/{{.Architecture}}'
```

预期输出 linux/amd64。手动下载时必须取得全部四个分卷；import.sh 校验每个分卷及合并归档后才导入。

环境不含模型权重和 FAIR-CSAR 数据。GPU 运行仍需宿主 NVIDIA 驱动和 NVIDIA Container Toolkit。
CPU 工程测试已通过；目标 GPU 前向/训练尚未实测。现有训练启动器仅接受 1/2 个进程，四张 4090 的配置适配不包含在此次镜像目录整理中。
