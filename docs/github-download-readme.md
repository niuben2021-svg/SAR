Qwen3-VL-4B LoRA 离线环境、代码与中文指南。

镜像由本仓库 GitHub Actions 从官方 Python 基础镜像、PyPI CUDA wheels 和源码重新构建，具体依赖由构建时解析。它与云工作区 2026-10-23 镜像并非同一字节版本，下载后以此 Release 的 SHA256 文件为准。

需要下载全部 `sar-vlm-qwen3vl4b-image.tar.gz.partNN` 镜像分卷，以及 `sar-vlm-qwen3vl4b-source.tar.gz`、`qwen3vl4b-offline-guide.md`、所有校验文件。

GitHub 自动附带的 Source code.zip 不是指定的源码交付包；请下载名称明确的 sar-vlm-qwen3vl4b-source.tar.gz 附件。

Linux：
```bash
sha256sum -c SHA256SUMS
cat sar-vlm-qwen3vl4b-image.tar.gz.part* > sar-vlm-qwen3vl4b-image.tar.gz
sha256sum -c sar-vlm-qwen3vl4b-image.tar.gz.sha256
gzip -dc sar-vlm-qwen3vl4b-image.tar.gz | docker load
```

Windows：在下载目录打开命令提示符 CMD，把所有 part01、part02 等依次用 `copy /b part01+part02+... sar-vlm-qwen3vl4b-image.tar.gz` 合并（使用分卷完整文件名），然后 `certutil -hashfile sar-vlm-qwen3vl4b-image.tar.gz SHA256`，与 .sha256 对照。切勿单独解压分卷。

目标平台：麒麟 V10、x86_64、Docker 20.10.24、驱动 595.58.03、2×A100 40GB。环境包含 CUDA 版 PyTorch、Transformers、ms-swift、PEFT。项目源码单独挂载。

不含预训练权重、FAIR-CSAR 数据或已训练 adapter。完整 Qwen3-VL-4B-Instruct 权重需另行下载，宿主机需要 NVIDIA Container Toolkit。工程 CPU 测试不代表真实 A100 训练已成功；按指南先单卡前向、3 步短跑，再双卡训练。

本次创建独立交付分支，不合并 main。
