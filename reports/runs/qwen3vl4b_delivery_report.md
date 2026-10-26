# Qwen3-VL-4B 离线交付记录

Task ID: OFFLINE-02
Status: PARTIAL（GPU 训练验证尚未执行）

Changed files:
- `src/sar_vlm/models/architecture.py`：4B/8B dense、非量化架构检查；adapter 取消硬编码 8B。
- `src/sar_vlm/deployment/*`、`config.py`、`configs/*`：4B 默认下载/校验/离线路径与 LoRA 配置。
- `src/sar_vlm/cli/prepare_sft.py`：已审核分类 CSV 转 JSONL、group/图像字节跨 split 检查。
- `src/sar_vlm/training/offline_smoke.py`、`train.py`：单卡短跑、全训练集双卡显式启动及证据门槛。
- `deploy/Dockerfile.runtime`、`offline.env`、`requirements-runtime-actual.txt`：真实 CUDA 环境及实装版本。
- `scripts/offline/{build_runtime,download_model,run}.sh`、entrypoint、源码打包脚本。
- 新增/扩展架构、数据转换和训练门槛测试；新增 4B 使用指南与 ADR-0002，更新工程依据。

Commands executed:
- `docker pull --platform linux/amd64 python:3.11-slim-bookworm`。
- 从官方 Python digest + PyPI CUDA wheels 实际 `docker build`，成功生成依赖镜像。
- `docker tag sar-vlm:cuda-runtime sar-vlm:qwen3vl4b`。
- `pytest -q`、ruff check/format、bash -n、git diff --check。
- 官方 `HfArgumentParser(SftArguments)` 校验配置及完整短跑/训练计划。
- 实际 `docker run --network none --read-only`，源码只读挂载，/tmp 和 /outputs 使用 tmpfs
  完成 CPU 数据链、60 项工程测试、pip check、两种训练计划解析；没有加载模型权重。
- Docker 客户端使用 API 1.41 完成上述容器运行，服务端仍是 28.4.0，不能等同实测 Docker 20.10.24。

Tests:
- 宿主：passed 60 / failed 0 / skipped 1（GPU）。
- 最终 CUDA 环境断网：passed 60 / failed 0 / deselected 1（GPU），2.93 秒。
- pip check、官方 Qwen/Swift 类导入、训练及推理参数解析：PASS。
- 真实预训练模型 GPU 前向、训练、DDP、SAR 识别评测：NOT RUN。

Actual hardware/software:
- 实际云环境：Linux x86_64、无可见 GPU、Docker 28.4.0、vfs、32 GB 临时文件系统。
- 镜像：Python 3.11.17，torch 2.8.0+cu128，torchvision 0.23.0+cu128，CUDA 12.8，
  Transformers 4.57.6，ms-swift 4.6.0，PEFT 0.20.0，qwen-vl-utils 0.0.14。
- torch 编译目标含 sm_80；实际 GPU count=0，不代表已经在 A100 运行。
- 用户报告目标：麒麟 V10、x86、Docker 20.10.24、驱动 595.58.03、2×A100 40GB。
- Python base digest: `sha256:f06d01d4ba3f8ad5af45a3ec87df19c1d317d8e8961edc0fabda61811af6f7cf`。
- 最终 image ID: `sha256:049d1b66c89fad1d69a282968e765c098c8a26c9323782794bdd74334a19551f`。
- 镜像架构 linux/amd64，Docker 报告未压缩大小 8,192,886,487 字节。

Measured GPU peak memory (if run): NOT RUN / null。

失败及处理：
- 官方 PyTorch 大镜像拉取未完成，停止该次拉取，改用官方 Python 基础镜像 + PyPI CUDA wheels。
- 依赖构建成功后追加镜像 ENV 时，vfs 深拷贝耗尽空间。移除该次失败中间容器，保留成功的依赖镜像；
  离线参数交由 deploy/offline.env 和启动脚本设置，无需另建巨大层。
- 首次容器工程测试 24 passed / 1 failed / 35 errors / 1 deselected，错误为 Errno 28 磁盘不足。
  改为 tmpfs 临时目录后断网重跑全部通过；不删除原失败日志。
- 官方模型下载端点返回 403，未获取权重，没有用结构 fixture 冒充真实模型。

Open risks/next dependency:
- 内网必须验证 NVIDIA Container Toolkit；宿主组件不能由本镜像替代。
- 原始 FAIR-CSAR 文件/标签未提供，不能声称已实现该数据发布格式的专用转换。
- CSV 转换为类别问答；物理 Adapter、检测框监督、评估系统属于后续任务。
- LoRA 学习率、token 预算、冻结视觉侧等是初始配置，未证实最优或保证不 OOM。
- 训练日志当前未聚合各 rank 峰值显存；评测和测试集保持独立。
- 无合并、无推送、无大规模训练。

Artifacts: 见本报告下方最终打包和传输校验记录。

最终打包和传输校验：
- `outputs/distribution/sar-vlm-qwen3vl4b-image.tar.gz`：4,606,175,178 字节，约 4.29 GiB / 4.61 GB。
- 镜像归档 SHA256：`7b90395db3f39bcb0bec914b67b599d2643ba5921ffd3bb9dc300d720a612818`。
- 实际 `docker save | gzip` 完成，SHA256 校验 PASS，实际 `gzip -dc | docker load` 返回
  `Loaded image: sar-vlm:qwen3vl4b`。导出与导入使用 API 1.41；服务端为 Docker 28.4.0。
- 源码包：`outputs/distribution/sar-vlm-qwen3vl4b-source.tar.gz`，另附同名 `.sha256`。
- 使用说明：`docs/qwen3vl4b-offline-guide.md`。
- 框架及依赖在环境镜像中，项目源码单独挂载。两个包均不含模型权重或真实数据集。
- `deploy/requirements-runtime-actual.txt` 与镜像内 pip freeze 一致。

环境镜像和项目代码交付完成。整体状态保留 PARTIAL，因为目标麒麟主机、真实模型、真实数据的
GPU 前向/训练尚未执行。不能将本记录视为 SAR 模型精度或微调成功的实验结果。
