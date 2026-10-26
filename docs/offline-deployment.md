> 本文为 2026-10-22 的 8B 历史交付说明。当前 4B 包请使用 [4B 内网指南](qwen3vl4b-offline-guide.md)，不要混用旧标签和路径。

# 外网构建、离线转运、内网运行

目标：先在内网完成真实 Qwen3-VL-8B 单卡图文前向，再进行明确限定步数的 LoRA 冒烟。FAIR-CSAR V1.0 已由用户下载；本仓库尚未接触其实际文件、许可、变量名或标注，不能声称转换和训练已经完成。

## 官方下载入口

- Qwen3-VL-8B-Instruct：[Hugging Face](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)、[ModelScope](https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct)。下载完整 snapshot，不只下载几个 safetensors。
- 模型官方代码：[QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)。本项目通过 Transformers 调用，无须再启动一个独立 Qwen 服务。
- 训练框架：[modelscope/ms-swift](https://github.com/modelscope/ms-swift)、[官方 Qwen3-VL 实践](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Qwen3-VL-Best-Practice.md)。镜像使用 ms-swift 4.6.0、Transformers 4.57.6；框架已经装进镜像，内网不再 pip install。
- 基础运行环境：[PyTorch 官方镜像](https://hub.docker.com/r/pytorch/pytorch)、[PyTorch 安装说明](https://pytorch.org/get-started/locally/)。脚本默认官方 `2.8.0-cuda12.6-cudnn9-runtime`，拉取后按实际 digest 构建，保留其配套 torch/torchvision。
- 宿主机 GPU 容器支持：[NVIDIA Container Toolkit 安装说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)、[CUDA 驱动兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/)。Docker 不包含宿主机内核驱动；宿主机 Docker、驱动、Toolkit 需要另行准备与操作系统匹配的离线安装包。

## 准备条件

双卡配置为 `configs/qwen3vl_lora_2xa100.yaml`：每卡batch1、累积8、有效batch16。保留四卡文件仅供原计划对照，本次不按四卡启动。驱动595.58.03属于较新驱动，可优先验收CUDA12.6镜像，但实际GPU容器请求仍需通过。

内网目标由用户于2026-10-22确认：麒麟 Kylin V10、x86、NVIDIA 驱动595.58.03、两张 A100 40GB。使用 Linux amd64 镜像；镜像内 Ubuntu/PyTorch 用户态不要求宿主机也是 Ubuntu。宿主机麒麟的内核、驱动、Docker 与 NVIDIA Container Toolkit 仍必须兼容。外网构建机可以没有 GPU，但应能运行 Linux amd64 Docker；推荐 Linux x86_64，Windows 可用 WSL2 的 Linux Docker。Apple Silicon 必须构建 amd64，不能把 arm64 镜像直接导入 x86_64 服务器。

外网机器为代码、CUDA 镜像、模型及导出文件预留约80–100GB空间（不含 FAIR-CSAR）；模型精确体积以下载为准。使用 CUDA12.6 基础镜像前对照内网实际驱动及 NVIDIA 兼容表；不在脚本中升级宿主驱动。若服务器需要不同组合，设置 SAR_BASE_IMAGE，重新构建并实测，不沿用 CPU 依赖锁。

内网可运行只读诊断脚本 `bash scripts/offline/check_kylin_host.sh /path/to/kylin-host-report.txt`，或先记录：

```bash
uname -m
cat /etc/os-release
nvidia-smi
nvidia-smi topo -m
docker version
docker info --format '{{json .Runtimes}}'
```

应确认 Docker 的 NVIDIA runtime 可用；麒麟宿主机应按实际发行版准备Docker/Toolkit离线包，不照搬Ubuntu apt命令。`docker load` 成功本身不证明 `--gpus all` 能用。本文没有假设 NVLink 一定存在。

## 外网机器：解压代码并构建

代码交付包解压后为 `sar-vlm/`。也可直接使用当前完整工作目录；不要只打包 Git 已追踪文件而漏掉新代码。

```bash
sha256sum -c sar-vlm-offline-source.tar.gz.sha256
tar -xzf sar-vlm-offline-source.tar.gz
cd sar-vlm
bash scripts/offline/build.sh
```

脚本会拉取基础镜像、解析不可变 digest、安装模型与训练框架、运行无 GPU 测试、完成 `--network none` CPU 验收。没有下载模型、没有训练。可配置：

```bash
SAR_BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime \
SAR_IMAGE_TAG=sar-vlm:offline bash scripts/offline/build.sh
```

镜像保留 `/opt/sar-vlm/BUILD-ENVIRONMENT.txt` 与 `BASE-TORCH.txt`。构建会约束基础镜像原有 torch/torchvision，避免 pip 把 GPU 包换成 CPU 包。所有后续脚本默认标签 `sar-vlm:offline`；换标签时设置同样的 SAR_IMAGE_TAG。

## 外网机器：下载完整模型

创建一个空的模型目录。中国大陆网络可先选择官方 ModelScope；另一个选择是 Hugging Face。

```bash
mkdir -p transfer/models
docker run --rm \
  --mount "type=bind,source=$PWD/transfer/models,target=/downloads" \
  sar-vlm:offline download-model --source modelscope \
  --destination /downloads/qwen3vl8b
```

或（不同时执行到同一目录）：

```bash
docker run --rm \
  --mount "type=bind,source=$PWD/transfer/models,target=/downloads" \
  sar-vlm:offline download-model --source huggingface \
  --revision main --destination /downloads/qwen3vl8b
```

Hugging Face 下载前把分支解析为实际 commit；也可传入指定 commit。ModelScope revision 以该站的版本为准，默认 master，不把它误当 Hugging Face 的 commit。脚本记录来源及 requested/resolved revision，并生成 transfer_manifest.json；ModelScope 无可解析 commit 时明确记录 null，文件 SHA256 标识实际转运内容。

下载失败可以用相同源/版本/目录续传；换源或 revision 使用新目录，避免混合不同权重。下载不限制文件后缀，保留 tokenizer、processor、chat template 和所有权重分片。目录中必须是真实文件，不能只复制指向外部 Hub 缓存的符号链接。SHA256 证明转运前后字节一致，不替代下载来源和模型许可核验。

## 外网机器：导出镜像和模型

```bash
bash scripts/offline/export_image.sh sar-vlm:offline transfer/sar-vlm-offline-image.tar.gz
```

这里使用 `docker save`，包含镜像所有层和依赖；不要用 `docker export` 替代。代码已包含在镜像内，模型和数据通过独立只读目录挂载。

模型目录可直接复制；如果需要压缩转运：

```bash
tar --exclude='qwen3vl8b/.cache' -C transfer/models \
  -czf transfer/qwen3vl8b.tar.gz qwen3vl8b
(cd transfer && sha256sum qwen3vl8b.tar.gz > qwen3vl8b.tar.gz.sha256)
```

向内网转运：镜像 tar.gz 与 .sha256、模型目录/模型 tar.gz 与校验文件、代码源码包。FAIR-CSAR 已下载，保留在内网数据目录，不写入 Docker 镜像。

## 内网：校验和导入

以下 `/srv/sar` 仅是可修改的示例工作目录；在自己的授权目录创建即可。

```bash
mkdir -p /srv/sar/models /srv/sar/data /srv/sar/outputs
cd /path/to/sar-vlm
bash scripts/offline/import_image.sh /path/to/transfer/sar-vlm-offline-image.tar.gz
# 如模型以 tar.gz 转运，先在转运目录校验再解压：
(cd /path/to/transfer && sha256sum -c qwen3vl8b.tar.gz.sha256)
tar -xzf /path/to/transfer/qwen3vl8b.tar.gz -C /srv/sar/models
```

将 FAIR-CSAR 原始数据放到 `/srv/sar/data/FAIR-CSAR`（也可修改挂载目录），源数据/模型只读、所有结果写到 outputs。先用完全断网模式验收：

```bash
bash scripts/offline/run.sh /srv/sar/models/qwen3vl8b /srv/sar/data /srv/sar/outputs cpu
bash scripts/offline/run.sh /srv/sar/models/qwen3vl8b /srv/sar/data /srv/sar/outputs verify-model
bash scripts/offline/run.sh /srv/sar/models/qwen3vl8b /srv/sar/data /srv/sar/outputs check
```

verify-model 会对所有模型文件计算 hash，8B权重可能耗时较长。check 检查布局、实际 GPU/BF16、库导入与拓扑；尚不等于权重加载/推理成功。run.sh 使用 `--pull never --network none`，不会临时联网安装依赖。默认所有 GPU 可见；仅单卡推理时可设置 `SAR_GPUS=device=0`。

## FAIR-CSAR：先盘点，再按明确格式准备显示副本

```bash
bash scripts/offline/run.sh /srv/sar/models/qwen3vl8b /srv/sar/data /srv/sar/outputs \
  inspect-data --root /data/FAIR-CSAR --output /outputs/fair_csar_inventory.json
```

清单只统计文件后缀/大小，不自动解释 `.mat` 的变量、复数编码、旋转框、标签坐标或分类词表。复图像真实包含哪些信息，以 README/元数据为准。当前 Qwen 使用图像显示副本，不会直接消费复数矩阵，也没有实现相位识别/子孔径分支。

工具仅接受**已经确认的二维 complex 数组**：NPY、显式指定 key 的 NPZ/MAT v5、complex TIFF。不猜测实虚部分开存储的变量、通道、转置方式或 MAT v7.3/HDF5 结构。

```bash
# 将示例文件名和 MAT key 替换为该数据集说明中确认的值。
bash scripts/offline/run.sh /srv/sar/models/qwen3vl8b /srv/sar/data /srv/sar/outputs \
  preview-complex --input /data/FAIR-CSAR/example.mat --key DOCUMENTED_COMPLEX_KEY \
  --output /outputs/previews/example.png
```

变换为 `20log10(abs(z))` 的1/99.5百分位裁剪，再线性映射到8-bit；这是显示变换，不是标定后散射系数。原始文件不写入，保持轴顺序、宽高，并保存来源 hash、显示参数和输出 hash。若原图本来是8-bit PNG，可以直接做前向，无须此转换。

## 单卡真实前向

显示图生成在 outputs，已在容器内映射为 `/outputs/...`：

```bash
SAR_GPUS=device=0 bash scripts/offline/run.sh \
  /srv/sar/models/qwen3vl8b /srv/sar/data /srv/sar/outputs \
  forward --image /outputs/previews/example.png \
  --query '请描述这幅SAR图像中可观测的内容；证据不足时明确说明。'
```

成功要求进程退出0，`/srv/sar/outputs/mvp/forward_output.json` 标记 PASS/inference_executed=true，记录真实图像输入、设备、耗时、显存和权重转运清单 hash。失败退出非零并写 forward_failure.json。检查当前时间戳，不能把留存的旧成功文件当作本次通过。

这一步只验收图文前向；空 detections 不表示无目标，也没有对 FAIR-CSAR 计算准确率。正式目标定位仍需完成 verified 标注转换、检测/切片/回填与评测。

## 后续：最多10步的 LoRA 冒烟入口

只有真实前向通过、FAIR-CSAR 许可/标签检查完成、已制作 reviewed SFT JSONL 后使用。本次不会自动启动。原始 `.mat`/标签文件不能直接作为 ms-swift 数据输入；具体转换器需根据实际样例实现和验收。

通用 SFT 每行包括 messages/images，图像路径应是内网容器可见的 `/data/...` 或 `/outputs/...`，或相对 JSONL 目录。不要批量套用一个未经核验的回答。Grounding 的 objects 格式与坐标转换必须另行通过当前版本模板集成测试。

为真实已审核 JSONL 准备 review.json，填写实际值，不使用工程 fixture 冒充：

```json
{
  "source_dataset": "FAIR-CSAR 复图像目标数据集 V1.0",
  "annotation_status": "verified",
  "license": "填写已核验的真实使用许可",
  "reviewed_by": "填写实际审核者",
  "sft_sha256": "填写 reviewed_sft.jsonl 的实际 sha256sum"
}
```

先仅检查计划：

```bash
bash scripts/offline/run.sh /srv/sar/models/qwen3vl8b /srv/sar/data /srv/sar/outputs \
  train-smoke --dataset /data/reviewed_sft.jsonl --review /data/review.json \
  --processes 1 --steps 3 --limit 100
```

计划通过也不代表数据模板训练已通过。确认需要执行少量训练时，在同一命令末尾加 `--execute`。脚本要求真实 GPU 前向报告与该模型 transfer manifest 匹配；验证 reviewed JSONL 的 hash、图像路径和消息结构，最多使用500条、最多10步，默认单卡3步。单卡先通过，再使用 `--processes 2` 验收实际双卡。该入口不执行正式大规模训练。

产物在 `/outputs/training/smoke-*/`：数据子集、实际命令、config/data/review hash、原始训练日志与运行结果。只有进程成功且本次 trainer_state 记录达到请求步数，才标记 PASS；没有采集的峰值显存记录 null。

## 当前验收边界

CPU 工程测试与源代码交付可在本机验证；实际 Docker 构建结果、GPU前向、FAIR-CSAR 实际格式适配和训练结果必须分别记录。参见 `reports/runs/offline_delivery_report.md`，以其中真实状态为准。
