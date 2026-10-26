# OFFLINE-01 内网部署交付记录

Task ID: OFFLINE-01 / MVP-03..06
Status: PARTIAL（源码/CPU 工程检查完成；真实 GPU/数据/镜像构建分别列出）
更新时间：2026-10-22T03:56:27.202421+00:00；用户时区 Asia/Shanghai。

## 目标机器

用户于2026-10-22报告：麒麟 Kylin V10、x86、NVIDIA 驱动595.58.03、2×A100 40GB。本机未连接该服务器，以上不是 nvidia-smi 实测结果。双卡配置每卡batch1、累积8、有效batch16；先单卡前向和短跑，再双卡。原四卡配置保留为原方案参考。

用户已下载 FAIR-CSAR 复图像目标数据集 V1.0；本机尚未获得数据目录/README/标注样例，变量名、复数编码、标注坐标、类别映射与许可均待核对。不能声称已经完成该数据集适配或训练。

## Changed files

- deploy/Dockerfile、requirements-container.txt、entrypoint.sh、.dockerignore：完整模型/训练运行环境，默认断网，保留官方基础镜像的GPU torch/vision，代码包含在镜像中。
- scripts/offline/build.sh、export_image.sh、import_image.sh、run.sh、check_kylin_host.sh、package_source.py：外网构建、镜像save/load/校验、只读模型数据挂载、麒麟宿主检查与源码打包。
- configs/offline.yaml、qwen3vl_lora_2xa100.yaml、scripts/train_sft_2xa100.sh：实际双卡硬件和本地路径配置。
- src/sar_vlm/deployment/：官方模型完整snapshot下载、转运hash、模型布局、GPU/BF16预检。
- cli/inspect_dataset.py、prepare_complex_preview.py：只读数据盘点与明确二维complex数组的幅度显示副本，保留原始复数文件。
- training/offline_smoke.py：默认仅解析计划，显式execute才执行，限定1..10步与最多500条；要求审核清单/JSONL hash和相同模型的真实GPU前向证明。
- models/qwen_adapter.py：记录本地模型来源及transfer manifest hash，避免把不同权重的前向记录混用。
- tests/：模型转运完整性/路径穿越/缺分片、复数源保留与显示几何、数据review和前向门槛、双卡参数等工程测试。
- pyproject.toml、AGENTS.md、PROJECT_CONTEXT.md、TASKS.md、README.md、docs/offline-deployment.md、ADR-0001：实际硬件更新、依赖、使用步骤及当前边界。

## Commands executed / Tests

- python -m pytest -q：**passed54 / failed0 / skipped1**（真实GPU测试未启用，不能计入通过）。日志 offline_pytest.log。
- ruff check/format：通过；Bash脚本 bash -n：通过。
- 双卡配置YAML与当前ms-swift官方SftArguments Namespace检查：PASS；原生swift sft --help仍只输出辅助参数，记录INCOMPLETE_HELP；未启动训练。
- Docker hello-world --network none：PASS，证明本机Docker基础容器执行可用。
- 以hello-world代表镜像运行实际export_image/import_image脚本：docker save→gzip→SHA256校验→docker load通过；仅是镜像转运流程验证，不是CUDA/Qwen镜像验收。日志offline_transport.log。
- 官方PyTorch基础镜像tag已通过registry manifest核对，digest为sha256:dab81780fd94483b67b4b5679cc0024939b08e48540d39476d284cb29002ed69；压缩大层约3.456GiB。实际拉取时大层出现自动重试，已停止本次未完成的下载（客户端退出130）；日志offline_base_image_pull.log。不把镜像tag存在、网络重试或小镜像通过视为CUDA构建通过。
- 完整sar-vlm:offline CUDA镜像当前存在：False。本次源码交付不包含已构建的CUDA镜像；按build.sh在外网机完成构建，再save/load。云机无GPU，当前Docker使用vfs、总盘约32GB，不把源码测试冒充真实GPU部署。

源码包已在独立新目录解压，使用解压出的src与tests复测：54通过/1跳过。日志offline_source_package_test.log。

## Actual hardware/software

当前云环境Linux x86_64、Python3.12.14、CPU torch2.14.1+cpu、Transformers4.57.6、ms-swift4.6.0；可见CUDA GPU=0。Docker28.4.0、vfs。已安装复数数据工程测试依赖scipy/tifffile。镜像使用官方CUDA12.6/PyTorch2.8.0基础环境构建，具体安装快照由镜像内BUILD-ENVIRONMENT.txt记录；本机CPU锁不能用作目标CUDA环境锁。

Measured GPU peak memory (if run): NOT RUN / null。没有真实Qwen权重加载、GPU前向、反向传播、FAIR-CSAR指标或模型性能结论。

## Artifacts / Open risks / next dependency

源码tar.gz及SHA256由package_source.py生成，包含当前未提交代码和完整部署手册，排除Git认证、模型、原始数据、venv与outputs。

1. 外网机执行build.sh，确认镜像内模型类/训练框架导入与CPU断网检查通过，下载完整官方模型并验证转运hash。
2. 内网麒麟宿主准备与发行版匹配的Docker/NVIDIA Container Toolkit；宿主驱动与内核模块不包含在镜像tar内。
3. 导入镜像，执行只读挂载、verify-model、check，再用一张授权图像完成真实单卡forward。尚未完成前不能宣称环境可用于训练。
4. 取得FAIR-CSAR目录/README与标签样例，确认复数变量、轴、显示映射和标注语义；当前只支持用户明确选择的二维complex NPY/NPZ/MAT v5/TIFF显示转换，未实现该数据集专用converter或相位/物理分支。
5. 单卡前向通过、审核SFT数据就绪后，仅按需要显式运行少量LoRA步骤；双卡须在真实硬件上独立验收。

本次没有执行训练、修改系统驱动、发布站点、提交或合并代码。
