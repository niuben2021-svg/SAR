# Qwen3-VL-4B ARM64 / A100 / CUDA 12.6 离线使用说明

目标：Linux ARM64（aarch64/SBSA）服务器、麒麟 V10、NVIDIA A100、驱动 560.35.03、Docker 20.10.24。不是 Jetson/L4T 镜像。宿主需安装适用于 ARM64 和目标操作系统的 NVIDIA Container Toolkit。

[打开 ARM64 完整下载页](https://github.com/niuben2021-svg/SAR/releases/tag/qwen3vl4b-arm64-cu126-20261028)

Release 的 Assets 提供全部镜像分卷、配套源码 `sar-vlm-qwen3vl4b-arm64-source.tar.gz`、本说明、下载/导入脚本、校验文件、实际依赖清单和 CPU 测试日志。镜像不存储在普通 Git 目录中。构建和检查成功后才发布附件；若页面尚不存在，说明构建尚未成功发布。

## 已发布文件（2026-10-28）

原生 ARM64 构建成功，CPU 测试 60 passed / 1 GPU deselected。镜像共 3,618,078,157 字节，三个分卷。

| 文件 | 下载 |
| --- | --- |
| ARM64-GUIDE.md | [下载](https://github.com/niuben2021-svg/SAR/releases/download/qwen3vl4b-arm64-cu126-20261028/ARM64-GUIDE.md) |
| BUILD-INFO.txt | [下载](https://github.com/niuben2021-svg/SAR/releases/download/qwen3vl4b-arm64-cu126-20261028/BUILD-INFO.txt) |
| cpu-tests.log | [下载](https://github.com/niuben2021-svg/SAR/releases/download/qwen3vl4b-arm64-cu126-20261028/cpu-tests.log) |
| download.sh | [下载](https://github.com/niuben2021-svg/SAR/releases/download/qwen3vl4b-arm64-cu126-20261028/download.sh) |
| image-inspect.json | [下载](https://github.com/niuben2021-svg/SAR/releases/download/qwen3vl4b-arm64-cu126-20261028/image-inspect.json) |
| import.sh | [下载](https://github.com/niuben2021-svg/SAR/releases/download/qwen3vl4b-arm64-cu126-20261028/import.sh) |
| requirements-arm64-actual.txt | [下载](https://github.com/niuben2021-svg/SAR/releases/download/qwen3vl4b-arm64-cu126-20261028/requirements-arm64-actual.txt) |
| sar-vlm-qwen3vl4b-arm64-cu126.tar.gz.part01 | [下载](https://github.com/niuben2021-svg/SAR/releases/download/qwen3vl4b-arm64-cu126-20261028/sar-vlm-qwen3vl4b-arm64-cu126.tar.gz.part01) |
| sar-vlm-qwen3vl4b-arm64-cu126.tar.gz.part02 | [下载](https://github.com/niuben2021-svg/SAR/releases/download/qwen3vl4b-arm64-cu126-20261028/sar-vlm-qwen3vl4b-arm64-cu126.tar.gz.part02) |
| sar-vlm-qwen3vl4b-arm64-cu126.tar.gz.part03 | [下载](https://github.com/niuben2021-svg/SAR/releases/download/qwen3vl4b-arm64-cu126-20261028/sar-vlm-qwen3vl4b-arm64-cu126.tar.gz.part03) |
| sar-vlm-qwen3vl4b-arm64-cu126.tar.gz.sha256 | [下载](https://github.com/niuben2021-svg/SAR/releases/download/qwen3vl4b-arm64-cu126-20261028/sar-vlm-qwen3vl4b-arm64-cu126.tar.gz.sha256) |
| sar-vlm-qwen3vl4b-arm64-source.tar.gz | [下载](https://github.com/niuben2021-svg/SAR/releases/download/qwen3vl4b-arm64-cu126-20261028/sar-vlm-qwen3vl4b-arm64-source.tar.gz) |
| SHA256SUMS | [下载](https://github.com/niuben2021-svg/SAR/releases/download/qwen3vl4b-arm64-cu126-20261028/SHA256SUMS) |

[实际构建和测试日志](https://github.com/niuben2021-svg/SAR/actions/runs/40696853973)。全部小附件已公开下载并校验；大分卷 HTTP 200，GitHub 摘要与 SHA256SUMS 一致，未再次完整下载三个大分卷。GPU 前向/反传未运行。

## 与原版的区别

- 平台从 linux/amd64 改为 linux/arm64，镜像标签 `sar-vlm:qwen3vl4b-arm64-cu126`。
- Python 3.11；官方 ARM64 torch 2.9.1+cu126 / torchvision 0.24.1 CUDA 12.6 wheel，下载地址及 SHA256 固定在 Dockerfile 中。
- 截图建议的 torch 2.8.0+cu126 / torchvision 0.23.0 ARM64 对应 wheel 在官方索引中不存在，因此使用可获得的匹配版本，不使用 CPU wheel 冒充 CUDA 版。
- 沿用原 Python slim 基础镜像方法；CUDA 12.6 运行库由官方 PyTorch wheel 的 NVIDIA 依赖提供，不重复叠加 nvidia/cuda 基础镜像。宿主驱动不会被容器修改。
- 保留 transformers 4.57.6、ms-swift 4.6.0、PEFT 0.20.0。依赖实际版本见 requirements-arm64-actual.txt。
- 使用 SDPA，不额外安装 FlashAttention/DeepSpeed。增加 ARM64 架构检查启动器和单 A100 保守配置，已有数据处理和训练入口保持兼容。
- 模型权重与数据不包含在镜像中。CI 检查镜像架构、CUDA 编译依赖、官方 Qwen/Swift 导入和原工程 CPU 测试；不宣称已经运行真实 A100 前向或训练。

## 1. 外网下载

下载本目录 download.sh，Linux 外网主机运行（下载主机可以是 x86_64）：

```bash
bash download.sh ./sar-arm64-delivery
```

也可在 Release 页面手动下载全部 Assets；所有 `.part01`、`.part02` 等镜像分卷都需要，不能只取第一个。不要混用旧版 AMD64 分卷及校验文件。

将整个下载目录搬入内网。准备至少 40 GB 可用磁盘空间以容纳分卷、合并归档和 Docker 解压层；实际空间要求随构建产物大小变化。

## 2. 内网导入环境与源码

```bash
uname -m     # 应为 aarch64
nvidia-smi   # 应能看到 A100 和宿主驱动
bash ./sar-arm64-delivery/import.sh ./sar-arm64-delivery
mkdir -p ./sar-arm64-code
tar -xzf ./sar-arm64-delivery/sar-vlm-qwen3vl4b-arm64-source.tar.gz -C ./sar-arm64-code
cd ./sar-arm64-code/sar-vlm
```

预期导入后的架构检查输出 `linux/arm64`。导入脚本先校验附件与合并镜像的 SHA256，再执行 docker load。若报 GPU runtime/driver 错误，先修复宿主 ARM64 NVIDIA Container Toolkit，不能通过安装 x86_64 工具包解决。

## 3. 挂载自己的权重和数据

以下三个变量使用你机器上的绝对路径：

```bash
MODEL=/srv/models/Qwen3-VL-4B-Instruct
DATA=/srv/data/SAR
OUT=/srv/outputs/SAR
mkdir -p "$OUT"
bash scripts/offline/run_arm64.sh "$MODEL" "$DATA" "$OUT" cpu
bash scripts/offline/run_arm64.sh "$MODEL" "$DATA" "$OUT" verify-model
bash scripts/offline/run_arm64.sh "$MODEL" "$DATA" "$OUT" check
bash scripts/offline/run_arm64.sh "$MODEL" "$DATA" "$OUT" forward --image /data/preview.png --query '请描述这张SAR图像中的目标。'
```

`preview.png` 替换成真实数据中已正确转换的图像。权重需完整，含原交付工具生成的传输校验清单；首次权重下载和清单生成方法见源码中的 `docs/qwen3vl4b-offline-guide.md`。启动器保持网络关闭、源码/权重/原始数据只读挂载。

## 4. 数据准备和微调

数据 CSV 格式、复数 SAR 显示转换、标签审核、按 source_group_id 划分训练/验证/测试的具体步骤，沿用 `docs/qwen3vl4b-offline-guide.md`，将其中 `run.sh` 替换为 `run_arm64.sh` 即可。不得将复数数据直接当普通 RGB 图输入，也不要自行编造散射物理标签。

完成数据准备后，假设输出为 `/outputs/sft/train.jsonl` 和 `/outputs/sft/train.review.json`：

```bash
# 必须先完成上一节真实前向；第一次不加 --execute 只检查训练计划。
bash scripts/offline/run_arm64.sh "$MODEL" "$DATA" "$OUT" train-smoke \
  --config configs/qwen3vl4b_lora_arm64_a100.yaml --processes 1 \
  --dataset /outputs/sft/train.jsonl --review /outputs/sft/train.review.json --steps 3

# 计划无误后，在同一条命令末尾添加 --execute，实际训练 3 步。
# 短跑成功后，使用实际生成的 smoke 目录中的 run.json 路径启动有限步训练：
bash scripts/offline/run_arm64.sh "$MODEL" "$DATA" "$OUT" train \
  --config configs/qwen3vl4b_lora_arm64_a100.yaml --processes 1 \
  --dataset /outputs/sft/train.jsonl --review /outputs/sft/train.review.json \
  --smoke-report /outputs/training/smoke-实际目录/run.json --steps 100
```

以上训练入口默认只输出计划；由你确认后添加 `--execute` 才运行。真实 smoke 报告文件名以终端实际输出为准。若是双 A100，可在单卡短跑成功后使用原 `configs/qwen3vl4b_lora_2xa100.yaml --processes 2`。本 ARM64 交付不表示已验证四张 4090 的分布式训练。

## 5. 验证范围与故障信息

查看 Release 的 `cpu-tests.log`、`image-inspect.json`、`requirements-arm64-actual.txt` 和 `BUILD-INFO.txt`。目标机器的模型前向、反向、GPU 峰值显存和训练效果均需实际测量；CPU 测试成功不能代替 GPU 验证。

GPU 首次运行失败时保留 nvidia-smi、uname -m、Docker 版本、完整错误堆栈和 preflight 输出。无需升级或覆盖原 x86_64 镜像。
