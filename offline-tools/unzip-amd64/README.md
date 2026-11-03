# Debian 12 bookworm AMD64 unzip 离线安装

unzip 版本 6.0-28，架构 amd64。包含官方 Debian 二进制包和 libbz2-1.0 依赖；依赖容器现有 libc6 >=2.34，不替换 libc6，不更新 Docker 镜像。

[下载 ZIP 离线包](https://github.com/niuben2021-svg/SAR/releases/download/unzip-amd64-bookworm-20261103/sar-unzip-bookworm-amd64-offline.zip)

[完整附件、校验和测试日志](https://github.com/niuben2021-svg/SAR/releases/tag/unzip-amd64-bookworm-20261103)

安装到已运行的 Debian12 AMD64 容器，把“你的容器名”替换为实际名称。先用原镜像已有的 Python 解开安装包，不需要提前安装 unzip：

```bash
docker cp sar-unzip-bookworm-amd64-offline.zip 你的容器名:/tmp/
docker exec -u 0 你的容器名 python -m zipfile -e /tmp/sar-unzip-bookworm-amd64-offline.zip /tmp/unzip-offline
docker exec -u 0 你的容器名 bash /tmp/unzip-offline/unzip-amd64/install.sh
```

也提供 tar.gz：可用容器已有的 `tar -xzf` 解开，然后运行同一个 install.sh。整个安装不使用 apt 或网络，安装脚本拒绝架构/发行版不匹配的环境及已有包降级。

安装后使用：

```bash
unzip -t /data/dataset.zip
unzip -n /data/dataset.zip -d /outputs/extracted
```

`-t` 检查压缩包，`-n` 不覆盖已有文件。输出放在可写挂载目录。`--rm` 临时容器退出后安装会消失，需要在同一次会话内完成解压，或以后重新运行安装脚本。

包来源、版本、SHA256 见 packages.json，版权许可保留在原始 .deb 的 `/usr/share/doc` 中。test.zip/test.txt 仅是解压验证用的合成工程样例。
