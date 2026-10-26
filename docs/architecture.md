# SAR-VLM 工程接口与研究边界

依据：仓库 `AGENTS.md`、`PROJECT_CONTEXT.md`、`TASKS.md`。以下区分研究设计与本次实现，不依赖历史聊天。

目标是按自然语言查询在 SAR 原图中检出全部指定目标，识别基础/细粒度类别，提供可定位证据，并在证据不足时退回粗类或拒识。研究主线为**知识引导的初始感知、基于本体约束的后验验证、反馈驱动的动态优化**。H1 搜索预算分配、H2 条件物理原型、H3 竞争假设与证据选择、H4 抗冲突历史先验都是待验证假设，当前没有领域效果结论。

## 输入与输出

- 输入：SAR 幅度/强度影像 + 文本查询；可选成像元数据、历史阵位、专家知识。原图只读；metadata 未知使用 null，不能以 0 伪装已知角度。
- Canonical v0.1：`scene_id/source_image/width/height/image_type/source_group_id/split/query/objects/label_provenance`，外加 metadata、区域、证据、hash、license。JSONL 中相对图像路径相对**清单文件目录**解析；YAML 的相对路径相对 **YAML 所在目录**解析。
- 所有内部框为原图像素 XYXY，右下半开，`0 <= x1 < x2 <= width`。不在 reader、renderer 或 adapter 隐式归一化。
- 最终目标输出为候选框、基础/细类、原始分数、校准分数、证据及决策。MVP 仅输出真实前向 `raw_text`、输入形状/图像 hash/权重 revision/设备/耗时；`detections=[]`，`calibrated_confidence=null`。空检测数组不表示原图无目标。
- `synthetic_fixture` 仅检查工程，不能作为 SAR 真值、训练监督或识别效果。真实数据需审核许可与 provenance，并按 `source_group_id` 划分防止相邻 tile 泄漏。

## 模块依赖

| 模块 | 输入 → 输出 | 前置依赖/监督 | 本次状态 |
|---|---|---|---|
| SARReader + CanonicalSample | 文件/metadata → 原生像素、dtype、range、hash、mask、校验记录 | PIL 可读图像；显式来源标签 | 已实现并测 8/16-bit |
| Qwen adapter | 8-bit 显示副本+文本 → 真实生成文本及 trace | 官方 Qwen3-VL-8B 权重、processor、兼容 torch/transformers、单卡 BF16 | 接口已实现，真实加载/前向受阻 |
| ResultWriter | 原图+独立给定的框 → 新 PNG 红框 | 合法像素框；8-bit L/RGB/RGBA | 已实现；框外像素不变 |
| TileIndexer / 坐标转换 / merge | 原图/配置 → 原生分辨率 tiles、affine/有效域 → 回填/去重 | 数据 schema、坐标单测 | P1 后续实现 |
| Scene Query Head | 全局特征+查询/知识 → 区域软优先级 | 目标覆盖/场景监督、Object Head 基线 | 未实现；必须保留非典型区域兜底搜索 |
| Object Head | 原生 tile/共享视觉特征+查询 → 像素候选框 | 独立 verified 检测标注、检测损失 | 未实现；不能用 LM 文本框替代密集检测头 |
| Physical Adapter | ROI+metadata mask+条件真实原型 → support/conflict/unobservable/unknown 证据 | 成像元数据审计、真实原型、基础感知 | 未实现；幅度图不支持虚构相位/子孔径 |
| KnowledgeEncoder | 历史/本体/专家知识 → 来源、时间、配准约束下的先验 | KNOW-01 schema；防未来泄漏 | 未实现；MVP 拒绝非空未验证历史记录 |
| Evidence Selector / Reasoner | 候选假设+ROI+可观测证据 → grounded evidence、竞争结论 | 专家证据标注、基础 SFT；物理分支可关闭 | 未实现；不以生成解释证明忠实性 |
| Trust Head | logits+覆盖/冲突 → 校准/接受粗细类/拒识 | 独立验证集、OOD/负样本、校准与风险评测 | 未实现；校准分数保留 null |

实现顺序：P0 数据/模型前向 → DATA-01/02 与 B0/B1 → BASE-03/DET-01 → SCENE → KNOW/PHYS → EVI/TRUST → 集成与消融。各损失只在具备对应监督后激活；不提前创建空模块作为完成证据。

## 版本核对

2026-10-21 获取并核对以下官方页面；框架参数仍需实际版本帮助文本验证：

- https://github.com/QwenLM/Qwen3-VL ：Transformers >=4.57，`apply_chat_template` 多模态输入与生成裁剪流程。
- https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Qwen3-VL-Best-Practice.md ：官方 Qwen3VL 类、LoRA、freeze_vit/aligner、IMAGE_MAX_TOKEN_NUM 等。当前文档使用 `tuner_type`。

训练配置采用仓库建议的 rank16/alpha32、micro batch1、累积4、四卡 effective batch16、BF16、max_length2048。它们是起始值，不是最优实测值。选择 SDPA 和 DDP，不把 FlashAttention/DeepSpeed 编译作为 P0 前置条件。
