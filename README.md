> **当前入口（2026-10-23）**：用户首轮采用 Qwen3-VL-4B-Instruct + 双 A100 LoRA。请阅读 [4B 内网部署与微调说明](docs/qwen3vl4b-offline-guide.md)。环境镜像与代码分开交付，模型权重另行下载。以下原 MVP 说明含 8B 历史配置。

# SAR-VLM MVP

查询驱动 SAR 全图目标定位的第一阶段工程。研究接口与依赖见 [docs/architecture.md](docs/architecture.md)，验收依据为 [TASKS.md](TASKS.md)。压缩包 HTML 仅是预设交互演示，与本 Python 工程的真实模型前向分开。

**当前交付：CPU 数据读取、schema、像素不变红框、配置与测试；真实 Qwen3-VL-8B 前向尚未通过。** 无训练/检测指标或已训练 checkpoint。没有实现 Scene/Physical/Evidence/Trust 高级分支。

## 安装

Python 首选 3.11；本机实测 3.12.14。已有 checkout 可直接使用；云任务已隔离，无需额外 worktree。

```bash
cd /workspace/SAR
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

数据/渲染测试不需要 torch、权重或网络服务。真实前向需要另外安装与目标机器驱动兼容的 PyTorch（使用 PyTorch 官方安装方式），然后：

```bash
python -m pip install -e '.[model]'
```

本机没有 GPU，安装的是 **CPU 版** torch，不能将其当作四卡集群 CUDA 环境。不要在 GPU 机器上照搬 CPU 版本锁。`.venv` 与模型权重不进 Git。`requirements-cpu-lock.txt` 记录本机实际安装版本（包含 ms-swift）；仅用于相同平台 CPU 环境复现，GPU 机器需单独验证 CUDA 组合。

训练参数检查可安装 `python -m pip install -e '.[training]'`，本次版本与检查结果以 `reports/runs/environment.json`、`config_validation.json` 为准。训练命令本身不会由本仓库脚本启动。

## CPU 检查

```bash
python -m sar_vlm.cli.check_env --config configs/mvp.yaml
python -m sar_vlm.cli.smoke_data --config configs/mvp.yaml
python -m pytest -q
ruff check src tests
ruff format --check src tests
# 或运行 CPU 检查组合（GPU 测试 deselected，不算通过）
bash scripts/run_mvp.sh --cpu
```

输出：`reports/runs/environment.{json,md}`，`outputs/mvp/data_trace.json`，`synthetic_fixture.{png,jsonl}`、`boxed_fixture*.png`。fixture 只验证工程，不是 SAR 图像或训练真值。重复执行保留已有红框图；源图/manifest 若被修改则报错。JSONL 记录相对图像路径相对清单目录解析，YAML 相对路径相对配置目录解析。

读取支持 PIL 可解码的 8-bit/16-bit PNG/TIFF 等；不做 SLC、GeoTIFF 地理语义或大幅图流式处理。红框仅支持 8-bit L/RGB/RGBA，保留框外像素及原始文件。16-bit 可原生读取，但不能隐式压缩后声称像素不变。

自有数据：在新 YAML 中配置 `input_image` 与 `manifest`（canonical JSONL），保留 provenance/hash/许可。无真实数据时不要把生成 fixture 设为 verified。

## 真实 Qwen 前向（需要 GPU 和权重）

修改 `configs/mvp.yaml` 或复制为本地配置：`model_path` 为官方 ID 或本地只读权重目录，`revision` 在实验前固定为实际 commit。默认 `local_files_only: true`，不会偷偷下载权重；如获准从 Hub 下载，可显式设为 false。报告保存请求 revision 与可解析的实际 revision；`main` 本身不是不可变版本。

```bash
python -m sar_vlm.cli.smoke_forward --config configs/mvp.yaml \
  --image /ABS/PATH/TO/AUTHORIZED_IMAGE.png \
  --query '请描述这幅SAR影像中可观测的内容'
# 独立真实 GPU pytest（必须显式启用，不使用 mock）
SAR_RUN_GPU_SMOKE=1 SAR_IMAGE=/ABS/PATH/TO/AUTHORIZED_IMAGE.png \
  python -m pytest -q -m gpu
```

成功才写 `outputs/mvp/forward_output.json`；受阻写 `forward_failure.json` 并退出 2，程序缺陷退出 1。检查报告的时间戳/配置 hash 与退出码，不要把以前留存的成功文件误认成当前运行。适配器使用官方 `Qwen3VLForConditionalGeneration`，单卡 BF16/SDPA、图像 token 预算1024、最多新生成128 token，不把生成文字当作可靠检测框。

红框独立入口（JSON 文件内容如 `[[8,8,32,30]]`，必须注明框来源）：

```bash
python -m sar_vlm.cli.draw_boxes --image INPUT.png --boxes boxes.json --output NEW_OUTPUT.png
```

## 训练配置检查（不训练）

```bash
bash scripts/train_sft_4xa100.sh --help
bash scripts/train_sft_4xa100.sh --dry-run
```

该入口校验 YAML 字段及实际 `swift sft --help` 的参数名称。ms-swift 4.6.0 的辅助解析器提前截获 `--help`，本机原始帮助被记录为 `INCOMPLETE_HELP`；随后以官方 `HfArgumentParser(SftArguments).parse_args` 检查实际参数名、类型和 choices，通过返回0。此处仅解析 Namespace，不触发参数 post-init、模型加载或训练。缺少 CLI 返回2，校验失败返回1。参数检查不证明分布式训练、反向传播或数据模板兼容。始终标记 `TRAIN_NOT_EXECUTED`。完整启动脚本及已审核 SFT 数据加载检查留待真实单卡前向与数据就绪后补充，不启动大规模训练。四张40GB不能视为单张160GB。

## 后续

先在有权重和单卡 BF16 GPU 的机器完成真实前向，再用 verified SAR 数据检查 SFT/Grounding 模板，最后进行少量训练短跑。P0 完成后再实现原生 tile/回填、Object Head 和高级研究模块。报告中须区分通过、失败、跳过与未运行；当前没有 SAR 识别效果结论。

## 内网 Docker 交付

外网构建、官方模型下载、镜像导出/导入、FAIR-CSAR 复图像显示副本及单卡验收命令见 [离线部署手册](docs/offline-deployment.md)。入口为 `scripts/offline/`；模型和数据只读挂载，镜像包含代码与 ms-swift。FAIR-CSAR 的实际变量与标注格式尚待样例确认，不能直接将原始复数文件当作 Qwen 图像训练输入。
