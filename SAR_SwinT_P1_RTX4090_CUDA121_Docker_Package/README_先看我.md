# SAR Swin-T + P1–P5 FPN + Faster R-CNN：4×RTX 4090 / x86_64 / CUDA 12.1 离线Docker部署包

## 注意：这个ZIP不是已经构建完成的Docker镜像

**本交付是完整代码、Dockerfile、权重下载/构建/导出/校验/导入/训练脚本，不是GB级Docker镜像tar.gz。**
当前ChatGPT文件环境没有Docker和容器GPU，不能真实生成或验证GB级CUDA镜像，切勿对本ZIP执行docker load。

在有互联网、有Docker的x86_64机器（例如你现在的4×4090服务器）运行 `bash scripts/01_build_export_image.sh`，会真正构建镜像并导出
`dist/SAR_SwinT_RTX4090_CUDA121_Image.tar.gz`，这是可执行`docker load -i`导入的完整离线镜像。不会把系统GPU驱动打入容器。

## 1. 目标宿主机（截图确认）

- CentOS Linux 7 (Core)，x86_64，Intel Xeon Platinum 8352V，144逻辑CPU，约1TB RAM。
- 4× NVIDIA GeForce RTX 4090 (24GiB/卡)；驱动535.146.02；`nvidia-smi`显示 CUDA 12.2（驱动接口上限，**不是**主机实际安装的CUDA Toolkit版本）。
- Docker client 19.03.15，Docker Engine server 26.0.0；Docker数据盘7.2TB空闲（以截图为准）。
- 选用 `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime`，linux/amd64。CUDA 12.1最低驱动约530.30.02，与你的535驱动兼容。
- PyTorch 2.3.1 与TorchVision 0.18.1+cu121严格配套，固定NumPy1.26.4与Pillow10.4.0。
- Docker 容器内使用自带Linux库，无需升级宿主CentOS7的glibc；但需宿主已经部署并配置NVIDIA Container Toolkit。

## 2. 工程目录

```text
SAR_SwinT_P1_RTX4090_CUDA121_Docker_Package/
├── Dockerfile
├── .dockerignore
├── README_先看我.md
├── project/       # 原始Swin-T+P1-FPN Faster R-CNN代码；含PyTorch 2.3兼容微调
├── weights/       # 构建时下载Swin-T ImageNet权重
├── scripts/
│   ├── 00_preflight_server.sh
│   ├── 01_build_export_image.sh
│   ├── 02_import_offline.sh
│   ├── 03_verify_gpu.sh
│   ├── 04_prepare_dataset.sh
│   ├── 05_train_4x4090.sh
│   ├── 06_eval.sh
│   └── 07_real_sample_smoketest.sh
└── dist/          # 真实构建后生成的镜像tar.gz及SHA256
```

## 3. 在联网的4×4090服务器上执行（建议按顺序）

把ZIP解压到一处有空间的目录，进入解压后的工程根目录：

```bash
bash scripts/00_preflight_server.sh
bash scripts/01_build_export_image.sh
```

第二条命令自动下载 `swin_t-704ceda3.pth` 权重、拉取官方镜像、构建项目镜像、严格加载权重检查，然后通过`docker save | gzip`导出真实镜像。

最终得到：

```text
dist/SAR_SwinT_RTX4090_CUDA121_Image.tar.gz
dist/SAR_SwinT_RTX4090_CUDA121_Image.tar.gz.sha256
```

检查真实镜像：

```bash
docker image inspect sar-swin-p1:rtx4090-cu121 --format '{{.Architecture}}'
ls -lh dist/
```

镜像应为`amd64`。真实镜像通常是数GB压缩包，具体体积以构建结果为准。

## 4. 先做GPU兼容性验收

```bash
bash scripts/03_verify_gpu.sh
```

必须显示：4个GPU可见、CUDA可用、NMS CUDA通过、ROIAlign CUDA通过、Swin-T+P1检测模型前后向通过。

如果出现 `could not select device driver "" with capabilities: [[gpu]]`，表示还没有配置NVIDIA Container Toolkit。宿主的`nvidia-smi`正常并不保证Docker容器可访问GPU。

先确认：

```bash
docker info --format '{{json .Runtimes}}'
which nvidia-container-runtime || true
```

需要由服务器管理员配置官方NVIDIA Container Toolkit；不要通过 `--privileged` 或主机驱动库乱挂载来绕过。

## 5. 数据集训练（训练512原生切片，测试1024重叠滑窗）

数据目录保持原样：

```text
dataset/
├── trainval/{PNGImages,Annotations,SLCMats,METAXmls,KeyPoints}/
└── test/{PNGImages,Annotations,SLCMats,METAXmls,KeyPoints}/
```

也支持 `dataset/SL-TRAINVAL/trainval/...` 和 `dataset/SL-TEST/test/...`。

运行时数据集通过挂载读取，**不打入Docker镜像**，节省体积。训练权重、指标保存到宿主机输出目录，不会随容器删除丢失。

下面假定：
- 数据集 `/data/dataset`
- 输出目录 `/data/sar_output`

```bash
bash scripts/04_prepare_dataset.sh /data/dataset /data/sar_output
bash scripts/07_real_sample_smoketest.sh /data/dataset
bash scripts/05_train_4x4090.sh /data/dataset /data/sar_output 1
bash scripts/06_eval.sh /data/dataset /data/sar_output val
```

- **优先单卡batch=1**，4张4090总batch=4；24GB卡在P1高分辨率分支下更应谨慎。显存宽裕可调单卡batch=2。
- 训练命令采用 `torchrun --standalone --nproc_per_node=4`，并使用混合精度。
- `--tile 512`为**原始分辨率裁剪**，不是将1024压缩到512；保留直升机等小目标的像素尺寸。
- 训练划分只从`trainval`中取训练和验证，`test`完整保留。
- P1为真正的stride=2浅层分支；非从P2简单上采样。
- 目前模型是幅度图基线，没有加入SLC子孔径增强，便于后续公平对照。

## 6. 如果想导入另一台无网4090机器

完整镜像在有网服务器上生成后，把 `dist/*.tar.gz` 和相应 `.sha256` 文件拷到目标机，在目标机执行：

```bash
bash scripts/02_import_offline.sh dist/SAR_SwinT_RTX4090_CUDA121_Image.tar.gz
```

或：

```bash
sha256sum -c dist/SAR_SwinT_RTX4090_CUDA121_Image.tar.gz.sha256
docker load -i dist/SAR_SwinT_RTX4090_CUDA121_Image.tar.gz
```

## 7. 现有项目注意事项

- 以你的真实SAR PNG和标注跑通为准；我没有完整数据集，无法报告真实mAP。
- 源工程的验证/测试划分来自文件名分组启发式，务必校验同一采集批次不跨train/val/test。
- 目前使用 HBB水平外接框，非旋转框。
- 脚本保存每5轮和最后轮，暂未自动选择最优val epoch；正式评测要按验证集选择checkpoint。
- 自定义AP评测适合第一轮比较，正式论文建议使用标准COCO评测复核。
- 完成GPU验证才可以说实际运行成功。当前交付只做源码/脚本静态检查与打包，无法代替目标4090 GPU实测。

## 8. 常见错误

| 错误 | 排查 |
|---|---|
| Docker daemon未启动 / 权限不足 | 先解决宿主机Docker访问权限。 |
| 构建超时或pip无法下载 | 配置外网服务器的Docker Hub、PyTorch wheel站点和权重下载网络访问。 |
| 下载镜像tag失败 | 核对Docker Hub网络连接与代理。 |
| `torchvision::nms does not exist` | TorchVision和PyTorch不匹配；确认使用本Dockerfile的cu121 wheel。 |
| `no NVIDIA GPU/driver` | 检查 `nvidia-container-toolkit` 与 `--gpus all`。 |
| 4090 CUDA OOM | 降低每卡batch为1，保持512 native tile；清理GPU其他任务。 |
| CUDA版本冲突 | 不要在镜像内随意pip upgrade torch/vision；不要升级到CUDA12.4/12.6镜像而保持535旧驱动。 |
