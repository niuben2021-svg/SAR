# ADR-0002：首轮微调切换 Qwen3-VL-4B-Instruct

日期：2026-10-23。状态：用户明确要求，已批准执行。

原默认底座为 8B，用户本轮要求先以 4B 完成自己数据集的 LoRA 微调和离线交付。
当前默认改为未量化 Qwen/Qwen3-VL-4B-Instruct；8B 的架构兼容和旧配置保留作后续对照。
用户目标平台：麒麟 V10、x86_64、Docker 20.10.24、驱动 595.58.03、2×A100 40GB。

先采用 BF16、SDPA、语言侧 all-linear LoRA、冻结视觉编码器及 aligner；这是起始配置，
未证明是最优 SAR 适配方案。首轮检验图像及经审核标签数据链路，不声称实现物理理解模块。
替代方案为直接做 8B 或新增物理 Adapter，本轮不采用。

离线环境与源码分开交付，源码只读挂载，模型/原始数据也只读挂载。
权重单独传输，以便代码更新无需重新传输模型或 GPU 依赖。
环境使用 Python 3.11 + PyTorch 2.8.0 CUDA wheels + torchvision 0.23.0 +
Transformers 4.57.6 + ms-swift 4.6.0；完整实装版本以镜像内
/opt/requirements-runtime-lock.txt 为准。CUDA 库包含在镜像中，内核驱动在宿主机。
镜像以 docker save 格式导出，不使用 docker export，不依赖 OCI attestations。

测试能证明数据约束、参数和 CPU 工程路径；实际 A100 前向/反传/DDP 仍需内网实测。
