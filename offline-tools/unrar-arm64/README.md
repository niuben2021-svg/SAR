# ARM64 Docker 容器用 unrar 离线安装包

适用：本项目已交付的 `sar-vlm:qwen3vl4b-arm64-cu126`，容器内部 Debian 12 / arm64。麒麟是宿主系统，本包安装到容器内部，不安装到宿主。原 Docker 镜像无需更改。

unrar 版本：`1:6.2.6-1+deb12u1`。来自 Debian bookworm 官方 non-free 仓库，未修改二进制；下载来源、依赖和 SHA256 见 packages.json。版权/许可保留在原 .deb 内的 `/usr/share/doc/` 中。

包内包含 unrar、libstdc++6、libgcc-s1、gcc-12-base。安装不访问网络，不替换 libc6/NVIDIA 驱动；已有更高版本的包不会被降级。适用于原交付镜像，不适用于 AMD64 或 Jetson/L4T 镜像。

## 下载

[下载离线 ZIP](https://github.com/niuben2021-svg/SAR/releases/download/unrar-arm64-bookworm-20261029/sar-unrar-bookworm-arm64-offline.zip)

[完整下载页与测试日志](https://github.com/niuben2021-svg/SAR/releases/tag/unrar-arm64-bookworm-20261029)

## 安装到正在运行的容器

在宿主运行，`你的容器名` 换成 `docker ps` 显示的实际名字。以下不需要在容器中预装 unzip：

```bash
docker cp sar-unrar-bookworm-arm64-offline.zip 你的容器名:/tmp/
docker exec -u 0 你的容器名 python -m zipfile -e /tmp/sar-unrar-bookworm-arm64-offline.zip /tmp/unrar-offline
docker exec -u 0 你的容器名 bash /tmp/unrar-offline/unrar-arm64/install.sh
```

## 解压自己的数据

进入容器后，例如数据文件在已挂载的 `/data`，输出在可写的 `/outputs`：

```bash
unrar t /data/dataset.rar
mkdir -p /outputs/extracted
unrar x -o- /data/dataset.rar /outputs/extracted/
```

`t` 检查压缩包，`x` 保留目录结构，`-o-` 不覆盖已有文件。分卷 RAR 必须把所有分卷放在同一目录，从第一卷开始解压，例如 dataset.part1.rar；有密码时按终端提示输入，不把密码写进共享脚本。

原项目 run.sh/run_arm64.sh 使用 `docker run --rm`：退出时临时容器会被删除，安装也不会保留。可在同一次交互会话里安装并解压，把结果写入 `/outputs` 挂载目录；以后新的临时容器若还需要 unrar，再运行一次离线安装脚本即可。整个过程不需要重建或保存新镜像。
