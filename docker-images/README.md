# Docker 镜像下载

| 架构 | 下载目录 | 状态 |
| --- | --- | --- |
| Linux AMD64 / x86_64 | [linux-amd64](linux-amd64/) | 已发布；已核实归档内架构 |
| Linux ARM64 / aarch64 | [linux-arm64](linux-arm64/) | ARM64 A100 / CUDA 12.6；独立构建发布入口见目录 |

amd64 与 x86_64 是同一种 64 位 CPU 架构；arm64 是不同架构，不能直接复用 amd64 二进制镜像。
镜像大文件存于 GitHub Releases，目录提供下载入口、校验与导入说明。
