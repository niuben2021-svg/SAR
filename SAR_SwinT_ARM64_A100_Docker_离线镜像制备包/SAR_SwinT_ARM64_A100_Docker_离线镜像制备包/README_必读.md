# ARM64 麒麟 V10 + 双 A100 40GB：离线 Docker 镜像制备包

## 这次交付的究竟是什么？

这是**可下载的 Dockerfile + 一键生成离线镜像脚本 + SAR 全部训练代码**。

⚠️ **这个 ZIP 不是已经构建好的 Docker 镜像**，不能执行 `docker load -i 这个zip`。
当前文件生成环境是 x86_64，无 Docker 程序且不能访问外网，因此无法可靠构建/执行 arm64 CUDA 镜像；不会将纯脚本 ZIP 假称为实际镜像。

外网有 Docker 的电脑/服务器上可根据下文生成 `SAR_SwinT_ARM64_A100_CUDA126_Image.tar.gz`；这个 **tar.gz 才是可在内网执行 `docker load` 的完整镜像**。

## 硬件适配与选择依据

- 主机：麒麟 V10 (Tercel)，aarch64 / 鲲鹏920，glibc 2.28，Linux 内核 4.19.90。
- GPU：2 张 NVIDIA A100-SXM4-40GB。
- 驱动：560.35.03，`nvidia-smi` CUDA 12.6。
- 容器：Docker 20.10.1。
- 基础镜像：`nvcr.io/nvidia/pytorch:24.10-py3` **linux/arm64/SBSA**，Ubuntu 22.04 + CUDA 12.6.2 + NVIDIA PyTorch 2.5.0a0。该版本的 NVIDIA 驱动最低要求为 R560，与宿主机匹配。
- 不使用 x86 的 pytorch/pytorch 镜像，不使用仅适合 Jetson 的 `l4t-pytorch` 镜像。
- 注意：CUDA 驱动可用不等于 Docker GPU 容器运行时已经配置；宿主机需额外安装配置 NVIDIA Container Toolkit（`docker info` 和后续 GPU 检查脚本可检测）。
- 麒麟主机 glibc 2.28 不阻止容器使用 Ubuntu 22.04 自带 glibc；但最终运行兼容性还须上机检验，尤其 NVIDIA Container Toolkit 与特定内核。

## ZIP 结构

```text
SAR_SwinT_ARM64_A100_Docker_离线镜像制备包/
├── Dockerfile                  # ARM64 NGC PyTorch 24.10
├── .dockerignore
├── README_必读.md
├── project/                    # 全部训练/推理/评测源代码和GPU算子验证
├── weights/                    # 外网自动放入官方Swin-T权重
└── scripts/
    ├── 00_外网下载权重.sh
    ├── 01_外网生成完整ARM64镜像tar.gz.sh
    ├── 02_内网导入镜像.sh
    ├── 03_内网GPU环境检查.sh
    ├── 04_内网准备数据.sh
    ├── 05_内网双卡训练.sh
    ├── 06_内网评测.sh
    └── 07_内网网络和样本测试.sh
```

## 第一步：在有网的机器生成镜像（只需做一次）

推荐**有网的ARM64 Linux主机**（如能接入双A100更好），也可以在有Docker 20.10+和BuildKit的**x86_64 Linux机器**制备 ARM64 镜像。Dockerfile不执行RUN命令，跨架构仅组装现成ARM64镜像层和源码，避免x86机器执行ARM二进制文件。

解压ZIP后进入解压出来的顶层目录：

```bash
bash scripts/01_外网生成完整ARM64镜像tar.gz.sh
```

脚本会自动：
1. 下载 `swin_t-704ceda3.pth` 权重，放入最终镜像；
2. `docker pull --platform linux/arm64 nvcr.io/nvidia/pytorch:24.10-py3`；
3. `docker build --platform linux/arm64` 创建 `sar-swin-p1:arm64-cu126`，包含全部SAR代码+权重；
4. 检查镜像 Architecture 必须是 arm64；
5. `docker save | gzip` 输出 `SAR_SwinT_ARM64_A100_CUDA126_Image.tar.gz` 和 SHA256 校验文件。

外网Docker可能需要配置代理或NGC registry访问。Docker镜像很大，准备至少**几十GB的外网下载/镜像构建磁盘空间**，根据实际压缩大小准备内外网传输空间。CUDA镜像不是轻量几百MB文件。

**非常重要：** 有网x86机器只能完成ARM64镜像层的组装和导出，不能由此证明ARM64 GPU算子已在目标硬件运行。真正的验证需要内网ARM64 A100服务器运行检查脚本。

## 第二步：把镜像文件拷进内网

将下面两份文件及本工具包 ZIP（或解压后的 scripts 文件夹）复制至内网：

```text
SAR_SwinT_ARM64_A100_CUDA126_Image.tar.gz
SAR_SwinT_ARM64_A100_CUDA126_Image.tar.gz.sha256
```

内网终端 cd 到镜像存放目录，执行：

```bash
sha256sum -c SAR_SwinT_ARM64_A100_CUDA126_Image.tar.gz.sha256
docker load -i SAR_SwinT_ARM64_A100_CUDA126_Image.tar.gz
docker image inspect sar-swin-p1:arm64-cu126 --format '{{.Architecture}}'
```

必须显示 `arm64`。也可用本包 `scripts/02_内网导入镜像.sh`。

## 第三步：检查GPU容器运行时（非常重要）

```bash
docker info --format '{{json .Runtimes}}'
docker run --rm --gpus all --network none --shm-size=8g \
  sar-swin-p1:arm64-cu126 python check_env.py --require-cuda
```

应该显示：`ARCH: aarch64`、`GPU count: 2`、`TorchVision CUDA NMS`、`TorchVision CUDA ROIAlign`、`CHECK PASSED`。

若显示 `could not select device driver "" with capabilities: [[gpu]]`，说明缺 `nvidia-container-toolkit` 或 Docker runtime没有配置。**宿主机 nvidia-smi 正常不等于 Docker 已能用GPU。** 此时需管理员通过与麒麟 ARM64 / Docker 20.10.1 配套的离线软件包安装 toolkit，并配置 daemon 后重启 Docker。请不要盲目安装 x86_64 的 RPM/DEB 包，也不要用 `--privileged` 绕过问题。

如果 CUDA NMS / ROIAlign 测试报 `torchvision::nms does not exist`，则需要在**外网 ARM64** 环境内核对 / 修复与 NGC PyTorch 相匹配的 torchvision GPU 编译版本；不要直接 `pip install --upgrade torch torchvision`，会破坏镜像内的CUDA兼容配套。**该点尚未在目标双A100实测。**

## 第四步：准备数据集（不缩放原图）

假设内网 `dataset` 在 `/data/dataset`，输出和权重均已放入镜像：

```bash
mkdir -p /data/sar_output
bash scripts/04_内网准备数据.sh /data/dataset /data/sar_output
```

正确目录为：

```text
dataset/{trainval,test}/{PNGImages,Annotations,SLCMats,METAXmls,KeyPoints}/...
```

也支持 `dataset/SL-TRAINVAL/trainval/...`、`dataset/SL-TEST/test/...`。

`trainval` 按成像场景划分训练/验证；保留官方 `test` 不参与训练。

## 第五步：小样本GPU前后向检查

```bash
bash scripts/07_内网网络和样本测试.sh /data/dataset
```

## 第六步：双A100训练（单卡batch先用1）

```bash
bash scripts/05_内网双卡训练.sh /data/dataset /data/sar_output 1
```

内部等价于：

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  --data-root /data/dataset \
  --split /workspace/output/split.json \
  --swin-weights /workspace/weights/swin_t-704ceda3.pth \
  --tile 512 --epochs 24 --batch-size 1 --workers 4 \
  --out-dir /workspace/output/runs/swin_p1
```

- `--tile 512` 是将1024原图按原始分辨率裁剪，不是缩小原图；P1有效stride=2。
- 训练集大、显存富余时可将每卡batch调为2。
- 保存权重至宿主机 `/data/sar_output/runs/swin_p1`，容器删除也不会丢。
- `--workers 4` 代表每个GPU对应DataLoader有4个worker，总数约8；主机96核+765GiB内存足够，但实际受Docker与存储性能影响。
- 如果验证最终论文指标，不要从test上挑选阈值/epoch。

## 第七步：评估

```bash
bash scripts/06_内网评测.sh /data/dataset /data/sar_output val
# 选择模型后才做官方test评测：
bash scripts/06_内网评测.sh /data/dataset /data/sar_output test
```

## 已经检查 vs 尚未验证

- ✅ 本包使用已提供的SAR项目源码；检测网络为 Swin-T V1 + P1–P5 + Faster R-CNN。
- ✅ 文件层面Python语法编译和脚本语法检查。
- ✅ 制备脚本明确按linux/arm64构建，且未在Dockerfile中运行ARM程序。
- ⚠️ 当前环境无Docker程序且外网不可达，因此**未生成真实Docker镜像tar**、未下载NGC镜像、未确认NGC 24.10 ARM64层中的TorchVision CUDA算子实测效果。
- ⚠️ 内网运行须有NVIDIA Container Toolkit；仅有Docker和nvidia-smi截图不足以判定已安装。

官方参考：
- NVIDIA PyTorch 24.10 release notes：https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-10.html
- NVIDIA NGC PyTorch Registry：https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
- NVIDIA Container Toolkit：https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
