# Qwen3-VL-4B 内网 LoRA 微调使用说明

本包对应 2026-10-23 用户指定平台：麒麟 V10、x86_64、Docker 20.10.24、
NVIDIA 驱动 595.58.03、2×A100 40GB。实际镜像为 Python 3.11.17、PyTorch 2.8.0+cu128、CUDA 12.8、Transformers 4.57.6、ms-swift 4.6.0。当前基座为 **Qwen3-VL-4B-Instruct 原始精度版本**，
不要下载 Thinking、FP8、AWQ 或 GGUF 替代。本包不包含数据集和预训练模型权重。

环境镜像包含 ms-swift 微调框架、CUDA 版 PyTorch、Transformers、PEFT 及其依赖。
项目代码包提供数据准备、校验、真实前向、单卡短跑、双卡微调的入口。代码通过只读挂载使用，
无需在内网 pip 安装。镜像依赖的真实安装锁定清单在 `/opt/requirements-runtime-lock.txt`。

## 1. 需要搬入内网的文件

1. `sar-vlm-qwen3vl4b-image.tar.gz` 和同名 `.sha256`：Docker 环境镜像。
2. `sar-vlm-qwen3vl4b-source.tar.gz` 和同名 `.sha256`：源码、配置、测试和本说明。
3. 完整的 `Qwen3-VL-4B-Instruct` 模型目录：权重分片、索引、config、processor、tokenizer、chat template。
4. 你已下载的数据集，以及自己核验的标签。原始复数数据不能直接当作普通 PNG 使用。

建议宿主机为镜像导入、模型、数据和训练输出预留至少 60 GB 可用空间；这不是实测最低值，
大数据集和多个 checkpoint 另算。两张 40GB 卡不能视为单张 80GB 卡。

### 模型官方下载

- 国内：https://modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct
- Hugging Face：https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- 框架：https://github.com/modelscope/ms-swift

在外网机器导入本镜像、解压源码后运行：

```bash
cd sar-vlm
bash scripts/offline/download_model.sh /absolute/transfer/qwen3vl4b modelscope
# 或把 modelscope 换成 huggingface
```

该下载脚本需要正常的外网 Docker 网络。网络需代理时请按单位网络配置容器代理和可信 CA，
不要关闭 TLS 校验。下载成功后脚本生成 `download_source.json` 与 `transfer_manifest.json`。
模型在本云环境未下载成功；403 限制不是模型需要付费的结论。

如果已经从官方网站手动下载完整目录，可在外网/内网运行下列命令生成本地传输清单：

```bash
docker run --rm --network none \
  --mount "type=bind,source=$PWD,target=/opt/sar-vlm,readonly" \
  --mount type=bind,source=/absolute/transfer/qwen3vl4b,target=/model \
  -e PYTHONPATH=/opt/sar-vlm/src sar-vlm:qwen3vl4b \
  python -m sar_vlm.deployment.artifacts create /model
```

清单验证的是传输字节一致性，不等于验证权重的上游真实性。模型目录可直接复制，或者：

```bash
cd /absolute/transfer
tar -czf qwen3vl4b-model.tar.gz qwen3vl4b
sha256sum qwen3vl4b-model.tar.gz > qwen3vl4b-model.tar.gz.sha256
```

## 2. 内网导入与目录布局

推荐目录（可更改宿主机路径，容器路径保持脚本约定）：

```text
/srv/sar4/
  sar-vlm/               # 解压的源码
  models/qwen3vl4b/       # 完整模型，包括 transfer_manifest.json
  data/                  # 原始数据/明确生成的显示图/labels.csv
  outputs/               # 可写：SFT 文件、日志、LoRA checkpoint
```

在保存安装包的目录执行：

```bash
sha256sum -c sar-vlm-qwen3vl4b-source.tar.gz.sha256
mkdir -p /srv/sar4
tar -xzf sar-vlm-qwen3vl4b-source.tar.gz -C /srv/sar4
cd /srv/sar4/sar-vlm
bash scripts/offline/import_image.sh /absolute/transfer/sar-vlm-qwen3vl4b-image.tar.gz
mkdir -p /srv/sar4/models/qwen3vl4b /srv/sar4/data /srv/sar4/outputs
```

模型压缩包在解压前也应 `sha256sum -c`。将模型内容放入上述模型目录，避免多嵌套一层文件夹。
镜像导出使用 `docker save` 格式，可由 `docker load` 导入；本云环境的 Docker 版本与内网不同，
Docker 20.10.24 + 麒麟宿主机上的 GPU 运行尚未实测。

### 宿主机 GPU 前提

```bash
docker --version
uname -m
nvidia-smi
nvidia-smi topo -m
docker info --format '{{json .Runtimes}}'
docker run --rm --network none --gpus all sar-vlm:qwen3vl4b \
  python -c 'import torch; print(torch.__version__,torch.version.cuda); print(torch.cuda.device_count()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'
```

期望识别两张 A100。驱动版本按用户提供的 595.58.03，版本数值高于 CUDA 12.8 的最低驱动要求，
但仍需以上命令确认实际兼容。无需在宿主机安装同版本 CUDA Toolkit；宿主机必须有可工作的
NVIDIA 驱动和 **NVIDIA Container Toolkit**。

如果出现 `could not select device driver ... gpu`，需管理员安装适配麒麟/RPM 系统的 Container
Toolkit 并配置 Docker runtime。它属于宿主机组件，不能仅靠导入本镜像完成：
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
不要在麒麟宿主机直接运行 Ubuntu apt 安装指令，也不要自动重装已有驱动。

## 3. 通用运行方式

之后所有命令均在 `/srv/sar4/sar-vlm` 下执行。定义一个便于复制的 Bash 函数：

```bash
sar4() {
  bash scripts/offline/run.sh \
    /srv/sar4/models/qwen3vl4b \
    /srv/sar4/data \
    /srv/sar4/outputs "$@"
}
sar4 cpu
sar4 verify-model
sar4 check
```

启动脚本从 `deploy/offline.env` 加载离线和缓存配置，`/tmp` 默认使用 2GB 内存临时目录（可通过 `SAR_TMP_SIZE` 调整）。所有常规运行使用 `--network none`，无需连接 Hugging Face、ModelScope 或日志云平台。
源码挂载在 `/opt/sar-vlm`，模型在 `/models/qwen3vl4b`，数据在 `/data`，输出在 `/outputs`。
前三者只读。命令内的图片/JSONL 路径使用容器路径，不能写 Windows 路径或宿主机 `/srv/...`。

## 4. 用自己的数据准备微调样本

### 4.1 FAIR-CSAR 复图像先生成明确的显示副本

先查看数据格式，软件不会猜测 MAT 变量或实虚通道：

```bash
sar4 inspect-data --root /data --output /outputs/dataset_inventory.json
# 示例：二维 complex NPY；MAT v5 或 NPZ 需明确 --key 变量名
sar4 preview-complex --input /data/raw/sample.npy --output /outputs/previews/sample.png
```

此转换采用幅度 `20*log10(abs(z))` 和 1/99.5 百分位显示映射，保存处理记录。
这是未标定的幅度显示图，不能声称保留全部复相位、真实散射机理或实现物理 Adapter。
MAT v7.3、特殊二进制、实虚双通道等格式需按实际数据说明另写读取器。

可以将经核对的显示图放入 `/srv/sar4/data/images/`。如果显示图留在 outputs，下一步的
`--image-root` 可以改为 `/outputs/previews`。不要修改原始复图像。

### 4.2 图像分类数据：CSV 转 JSONL

在 `data/labels.csv` 填写你核验的真实标签。可从 `configs/dataset_labels.template.csv` 复制表头。必需字段：

```csv
image,label,source_group_id,split,annotation_status
```

每一行提供：相对 `image-root` 的图片路径、真实类别、目标/采集组 ID、`train/val/test`、`verified`。
同一目标的相邻视角、相近切片应使用同一 source_group_id，并在同一 split。
程序拒绝同组跨 split 和逐字节相同图像跨 split；不能自动识别所有近似图像泄漏。
不要把文件名猜测或未审核标签写成 verified。

可选 `question` 自定义问题、`physical_observation` 提供实际观测信息。它们是输入文本，
类别字段是期望输出；没有可靠的解释标签时不会生成所谓散射解释真值。

```bash
sar4 prepare-sft \
  --csv /data/labels.csv --image-root /data/images \
  --output /outputs/sft-v1 \
  --reviewed-by '你的审核人姓名或标识' \
  --source-dataset 'FAIR-CSAR复图像目标数据集V1.0' \
  --license-note '填写你已核对的数据使用许可或授权范围'
```

输出 train/val/test JSONL、review 文件和 data_manifest.json。review 记录用户明确的审核声明，
不是程序自动验证标签正确。输出目录已存在时拒绝覆盖。

### 4.3 已有图文问答数据：直接使用 JSONL

ms-swift 格式每行一个 JSON。结构示例（必须把路径和回答替换为真实内容）：

```json
{"messages":[{"role":"user","content":"<image>请识别该SAR图像的目标类别。"},{"role":"assistant","content":"填写人工核验的类别"}],"images":["/data/images/实际文件.png"]}
```

支持多张图，`<image>` 数量应与 images 一致。图片必须存在、可读且已做明确的 uint8 显示转换。
如果人工编写 JSONL，也需同目录 review.json：

```json
{"annotation_status":"verified","source_dataset":"填写数据来源","license":"填写实际许可","reviewed_by":"审核人","sft_sha256":"填写JSONL文件的SHA256"}
```

用 `sha256sum train.jsonl` 获取真实摘要；修改任何 JSONL 字节后必须更新审核记录。
当前直接 JSONL 入口不自动验证目标分组划分，使用者必须保证 train/val/test 独立。
当前 CSV 转换器面向类别问答；若做检测/细粒度证据定位，需额外核验框坐标和监督格式。

## 5. 先完成真实单卡前向

换成你自己的真实显示图路径：

```bash
SAR_GPUS=device=0 sar4 forward \
  --image /data/images/实际文件.png --query '请识别图像中的目标类别。'
```

成功写入宿主机 `outputs/mvp/forward_output.json`，失败写 forward_failure.json 并返回非零。
核对本次命令退出码和报告时间，避免把旧成功报告当成本次结果。
输出文本只证明模型执行，不能直接当成识别准确率。

## 6. 单卡 3 步微调短跑

先检查计划，不执行训练：

```bash
SAR_GPUS=device=0 sar4 train-smoke \
  --dataset /outputs/sft-v1/train.jsonl --review /outputs/sft-v1/train.review.json \
  --processes 1 --steps 3 --limit 100
```

确认数据和计划后自行加 `--execute`：

```bash
SAR_GPUS=device=0 sar4 train-smoke \
  --dataset /outputs/sft-v1/train.jsonl --review /outputs/sft-v1/train.review.json \
  --processes 1 --steps 3 --limit 100 --execute
```

该命令最多允许 10 步和 500 条样本，使用自己的已审核样本。
每次生成独立 `/outputs/training/smoke-*/`，其中有 run.json、train.log、reviewed_subset.jsonl
和 checkpoints。成功必须实际产生 trainer_state.json 且步数达标；计划通过不算训练成功。
日志保存至文件，另开终端 `tail -f /srv/sar4/outputs/training/smoke-实际ID/train.log` 查看。

## 7. 双卡 LoRA 微调

单卡短跑成功后，把下面 `smoke-实际ID` 替换成刚成功运行的目录名：

```bash
sar4 train \
  --dataset /outputs/sft-v1/train.jsonl --review /outputs/sft-v1/train.review.json \
  --smoke-report /outputs/training/smoke-实际ID/run.json \
  --processes 2 --steps 100
# 上面只验证计划。准备执行时使用相同命令加 --execute：
sar4 train \
  --dataset /outputs/sft-v1/train.jsonl --review /outputs/sft-v1/train.review.json \
  --smoke-report /outputs/training/smoke-实际ID/run.json \
  --processes 2 --steps 100 --execute
```

这是用户手动发起的训练入口。本次交付没有自动启动上述训练。
`train` 会读取全部训练 JSONL，按显式 steps 截止，不会因 YAML 中 epochs=1 提前停止；
max_steps 优先，样本少时会重复迭代。先用短预算验证，再根据数据量和验证集表现选择训练预算。
两卡 DDP 下每卡 micro-batch=1、梯度累积=8，有效批量=16；单卡时为8。

配置文件：`configs/qwen3vl4b_lora_2xa100.yaml`。起始设置为 BF16、SDPA、LoRA rank16/alpha32、
学习率1e-4、max_length2048、图像预算1024token、gradient checkpointing。
冻结 ViT 和 aligner，LoRA 作用在语言侧线性层；尚未比较视觉侧适配的收益。
修改配置后建议重新做短跑。GPU 显存需求仍需 A100 实测，不能保证所有分辨率/序列组合都不 OOM。
OOM 时优先降低图像预算/序列长度；不要通过把复图像原始内容覆盖成低精度文件处理。

run.json 记录命令、数据/配置/权重清单摘要、软件/硬件、步数和退出码。当前启动器没有汇总
训练各 rank 的 PyTorch 峰值显存，该字段为 null；不能据此填入“实测显存”。
当前训练入口不自动做验证集评估，也不输出准确率。val/test 文件单独保留，不能参与训练。

## 8. 使用训练出的 LoRA

结果目录在 `/outputs/training/train-*/checkpoints/checkpoint-步数/`，实际以目录内容为准。
其中的 adapter_model.safetensors 是 LoRA 增量，不能脱离原始 4B 模型单独使用。
保存基座目录、配置、数据清单和 adapter 一起归档。

交互式验证可运行（目录名替换为实际路径）：

```bash
SAR_SHELL_GPU=1 SAR_INTERACTIVE=1 sar4 shell -c 'swift infer --model /models/qwen3vl4b --adapters /outputs/training/train-实际ID/checkpoints/checkpoint-100 --infer_backend pt --stream false'
```

若需要交互终端，在直接 docker run 中加 `-it`，或使用下面的 `--val_dataset` 批量方式。
推荐用独立 test.jsonl 记录预测，并人工/脚本按目标类别计算指标：

```bash
SAR_SHELL_GPU=1 sar4 shell -c 'swift infer --model /models/qwen3vl4b --adapters /outputs/training/train-实际ID/checkpoints/checkpoint-100 --infer_backend pt --stream false --val_dataset /outputs/sft-v1/test.jsonl --result_path /outputs/test_predictions.jsonl'
```

shell 模式默认不申请 GPU；上述推理请使用手动 GPU 命令，或给 run.sh 设置 `SAR_SHELL_GPU=1`。
这些是部署后的使用步骤，本次没有实际训练 adapter 或运行模型评测。

## 9. 外网自行重建镜像

如果需要重新构建或本次镜像下载不可用，在有网络的 Linux x86_64 Docker 主机执行：

```bash
cd sar-vlm
bash scripts/offline/build_runtime.sh
bash scripts/offline/export_image.sh sar-vlm:qwen3vl4b transfer/sar-vlm-qwen3vl4b-image.tar.gz
```

使用 Python 官方基础镜像，解析到 digest，再安装固定核心依赖；完整依赖版本由构建后 pip freeze
保存。依赖列表不是完整可重现锁，未来重建可能解析到不同间接依赖；本次导出的镜像字节才是
本交付的固定环境。镜像内的实装清单可用于后续受控重建。

`build_runtime.sh` 默认使用宿主机 `/etc/ssl/certs/ca-certificates.crt`；不同发行版可用
`SAR_BUILD_CA=/path/to/trusted-ca-bundle.pem` 指定。仅接受有效可信 CA，不使用跳过证书校验选项。
