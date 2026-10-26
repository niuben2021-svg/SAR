# AGENTS.md — SAR-VLM / Codex 工程执行规范

> **2026-10-23 用户已确定更新**：首轮底座改为 `Qwen/Qwen3-VL-4B-Instruct`，BF16 LoRA；内网 Docker 20.10.24、麒麟 V10、x86、驱动 595.58.03、2×A100 40GB。下文原 8B/四卡要求作为历史设计参考，当前实现以 `docs/ADR-0002-qwen3vl4b.md` 和 `docs/qwen3vl4b-offline-guide.md` 为准。

> 适用对象：Codex及所有自动化编码代理、软件研发成员。版本0.1，2026-10-19。  
> **首要指令：按本文件 + `PROJECT_CONTEXT.md` + `TASKS.md` 执行；先实现MVP，不为追求宏大架构而跳过可运行的验证链路。**  
> 标记：**【已确定】**固定约束；**【建议尝试】**可替换方案；**【尚需验证】**仅为假设，不能写成结果。

## 1. 项目边界与任务优先序

**【已确定】项目用途**：研究SAR影像中由自然语言查询驱动的全图目标定位、细粒度识别、物理散射与历史知识证据查验、可信拒识。应用对象飞机/车辆/舰船等，目标是可信、可复现、可评测的 SAR-VLM 原型，不是人工画框的自动包装器。

**【已确定，2026-10-22用户更新】实际目标计算平台**：2张NVIDIA A100，每张40GB；麒麟V10、x86、用户报告驱动595.58.03。原四卡文档/配置保留为设计参考，实际启动以双卡配置为准。优先`Qwen/Qwen3-VL-8B-Instruct`；首轮使用BF16 LoRA微调，以每卡显存40GB为上限估算和测试。训练使用真实集群情况校准，不能假定所有GPU之间必有NVLink。

Codex执行优先级：

1. 阅读`TASKS.md`中`P0/MVP`任务；在现有仓库上扫描结构、安装方式和已有代码。已有功能优先复用，不重写仓库。
2. 先完成`数据读取→模型加载→单次前向→训练配置→测试`，形成可运行闭环和运行记录。
3. 然后按依赖顺序实现canonical schema、tile回填、基线SFT、检测、Scene Head、Physical Adapter、Evidence/Trust。
4. 每个阶段必须附单元测试、运行命令和真实日志；未通过的事项明示`BLOCKED`或`NOT RUN`。**不得因缺少权重或数据就编造成功运行日志。**

## 2. 明确的技术决策与禁止事项

| 标签 | 硬性工程约束 | 原因 |
|---|---|---|
| MUST | 使用Qwen3-VL-8B开源底座起步，不从零训练8B | 现有项目训练效率和可复现性 |
| MUST | 总体是统一可训练模型结构，预处理/切片可工程化，不能改造成多智能体串行工具链 | 研究问题聚焦感知、证据及认知机制 |
| MUST | 大图用全局概览+原生分辨率tile/ROI | 下采样会丢失小目标 |
| MUST | 场景和历史只用于**软优先级**，不能作为硬排除规则 | 防止非常规目标漏检 |
| MUST | 密集目标定位须规划单独可训练Object Head，不能把LM文本框输出当唯一最终方案 | 避免生成式漏检/漂移 |
| MUST | 内部bbox统一原图像素XYXY（半开），转换集中在一个模块 | 避免回填和评测错位 |
| MUST | 幅度-only数据不提取/声称SLC相位、子孔径、真实极化矩阵 | 观测数据不支持 |
| MUST | 历史阵位带时间、配准误差及来源，历史不能直接生成当前检测真值 | 防止伪检出和时间泄漏 |
| MUST | 未经过校准的logits不能称为可信概率；未知类与拒识要有明确定义 | 可信评估要求 |
| MUST | 真实ground truth、弱标注、目视候选、synthetic fixture严格区分 | 禁止数据污染 |
| MUST NOT | 编造实验结果、GPU峰值、消融图、真值框、已训练权重 | 科研诚信 |
| MUST NOT | 用全图缩放到1024一次推理即宣称找全小目标 | 与既定方案冲突 |
| MUST NOT | 在MVP阶段加入仿真引擎、复杂RL证据选择、全量联合训练作为前置条件 | 违反分阶段研发 |

术语必须统一为**知识引导的初始感知、基于本体约束的后验验证、反馈驱动的动态优化**。报告论文时不要使用不经定义的笼统“高时效性”。

## 3. 建议的代码仓库结构

> 【建议尝试】新仓库可按如下结构创建。现有仓库应先适配，不得无条件覆盖。

```text
sar-vlm/
├── PROJECT_CONTEXT.md
├── AGENTS.md
├── TASKS.md
├── README.md
├── pyproject.toml
├── requirements-lock.txt           # 环境实测后生成；首日不伪造
├── configs/
│   ├── mvp.yaml
│   ├── qwen3vl_lora_4xa100.yaml
│   ├── tile_search.yaml
│   └── experiments/                 # 基线/消融的不可变配置快照
├── src/sar_vlm/
│   ├── __init__.py
│   ├── data/
│   │   ├── reader.py                # 只读图像及metadata
│   │   ├── schema.py                # 统一数据模型+验证
│   │   ├── tile.py                  # 切片、坐标变换
│   │   ├── conversion.py            # COCO/YOLO/SFT、官方Grounding转换
│   │   └── split.py                 # source_group级拆分
│   ├── models/
│   │   ├── qwen_adapter.py          # 仅处理官方Qwen模型接口
│   │   ├── scene_head.py
│   │   ├── object_head.py
│   │   ├── physical_adapter.py
│   │   ├── knowledge_encoder.py
│   │   ├── evidence_selector.py
│   │   ├── reasoner.py
│   │   └── trust_head.py
│   ├── training/
│   │   ├── losses.py
│   │   ├── config.py
│   │   └── runner.py
│   ├── inference/
│   │   ├── pipeline.py
│   │   ├── merge.py                 # 跨tile坐标/NMS/去重
│   │   └── result_writer.py         # 像素不变的红框叠加
│   ├── evaluation/
│   │   ├── detection.py
│   │   ├── search_budget.py
│   │   ├── evidence.py
│   │   └── calibration.py
│   └── cli/
│       ├── check_env.py
│       ├── smoke_data.py
│       ├── smoke_forward.py
│       ├── prepare_sft.py
│       └── draw_boxes.py
├── scripts/
│   ├── run_mvp.sh
│   ├── train_sft_4xa100.sh
│   └── eval_baselines.sh
├── tests/
│   ├── fixtures/                    # 小型合成图/合法数据schema
│   ├── test_reader.py
│   ├── test_schema.py
│   ├── test_tiles.py
│   ├── test_coordinates.py
│   ├── test_draw_boxes.py
│   ├── test_sft_converter.py
│   └── test_model_smoke.py
├── data/
│   ├── raw/                         # gitignore，不上传受许可约束原始数据
│   ├── annotations/
│   ├── processed/
│   └── manifests/
├── outputs/                          # 全部gitignore
└── reports/
    ├── runs/                         # 环境、运行日志、配置哈希
    └── audits/                       # 数据/结果审计
```

每个模块公开稳定的Python接口。`src/sar_vlm`采用相对稳定的包导入，避免脚本目录依赖 `sys.path.append`；不同外部框架的API放在adapter中统一处理。**不要提前创建大量空类/空文件作为完成任务的替代品**。

## 4. 数据和坐标硬契约

- Canonical字段详见`PROJECT_CONTEXT.md §4`。所有样本都应具有`schema_version`,`scene_id`,`source_image`,`width`,`height`,`query`,`objects`,`source_group_id`,`split`及provenance标记；如metadata不可知使用`null`，避免填0。
- 内部bbox统一`(x1,y1,x2,y2)`、原图像素、半开；坐标范围合法，框必须有正宽高。第一个版本仅要求HBB；OBB单独新增字段和任务，不能悄悄改已有XYXY含义。
- 局部tile与原图坐标转换必须双向测试：`global = local + offset`仅对未缩放情况成立；任何缩放和padding必须附affine及有效区域。
- 模型图像预处理副本与原图分离：**读取路径为只读，绝不原地覆盖**。红框可视化只允许输出到新文件；测试确认除线条像素外与原图一致（允许绘制框的颜色发生变化）。
- `ms-swift`官方Grounding模板支持绝对框经框架转换到Qwen3-VL相对1000坐标；只能在`conversion.py`处理该步骤，写明输入/输出单位与转换方向，确保最多一次归一化。
- 多模态SFT格式遵从所锁定版本的ms-swift：`messages`/`images`和可选`objects`等；必须用一个真实的、标签经核验的样例做转换集成测试。
- `raw_model_score`与`calibrated_confidence`是两个不同字段；未建立校准模型时，后者始终`null`。
- `annotation_status`可取`verified`,`weak`,`candidate`,`synthetic_fixture`,`illustrative_only`,`pending`；只有显式纳入的`verified`真值可用于核心测试指标。
- 禁止不同成像模式、真实/合成、训练/测试来源在无标记条件下混入同一统计口径。

## 5. 训练与模型工程约束

### 5.1 依赖与设备核查

- 首次运行记录：`nvidia-smi`、`nvidia-smi topo -m`、Python/PyTorch/CUDA/Transformers/ms-swift版本、GPU名称、可用显存、模型权重revision和git commit。
- 不擅自升级CUDA驱动、卸载系统包、清空checkpoint。若受集群权限限制，输出明确阻塞原因及可复制执行步骤。
- 外部模型下载可能需要网络/凭据/许可；支持本地`model_path`及只读权重缓存。如果环境不允许下载，仅完成CPU数据测试并把GPU前向标记`BLOCKED`，不能伪造一次推理。
- BF16首选，先确认PyTorch设备支持；FlashAttention只有在已安装且实际兼容时才启用。若不可用，优先使用PyTorch SDPA或官方默认实现，不能因FlashAttention编译失败阻塞MVP。
- 单GPU完成前向与低成本冒烟测试后再启动`NPROC_PER_NODE=4`的分布式微调。

### 5.2 训练配置与记录

- 【建议尝试】初始LoRA：`rank=16`,`alpha=32`、每卡micro batch1、梯度累积4、`max_length=2048`、`IMAGE_MAX_TOKEN_NUM=1024`、BF16、冻结ViT与aligner、gradient checkpointing，四卡DDP/ZeRO-2。只是**初始参数，不是已测最佳配置**。
- 只调用在当前环境由`swift sft --help`验证的参数；CLI和API可能随ms-swift版本变化。优先采用官方Qwen3-VL最佳实践而非复制其他Qwen版本的脚本。
- SFT训练先使用少量经核验的SAR图像/标签做100–500条冒烟训练；绝对禁止给synthetic fixture假标签做“真实检测效果”汇报。
- 每项训练需保存：配置快照、git sha、权重revision、数据清单hash、seed、命令、环境、峰值显存、日志、已运行步数、失败堆栈（如有）。
- 跨阶段新增损失必须具备明确监督来源、禁用开关、独立消融和测试；无标签loss权重保持0。

## 6. 代码质量要求

1. Python以3.11为首选，允许根据已安装CUDA/PyTorch兼容性修正；**严禁凭空锁定未验证兼容的torch/cu版本**。
2. 公共函数必须有类型提示、docstring（坐标单位/shape/是否修改输入）、明确的异常消息。
3. 用`pathlib`；避免硬编码个人目录；路径从CLI或YAML获取。
4. 图像必须标明`dtype`、强度范围和是否经过显示变换；大图读写考虑内存，MVP先支持PIL可读取类型，GeoTIFF/SLC后置。
5. 结构化输入输出优先dataclass/Pydantic（任选一种并统一），禁止到处依赖不受验证的散乱字典。
6. 使用`logging`，研究输出优先JSON/JSONL/CSV；日志不得打印敏感文件内容/Token/密钥。
7. 设置随机种子并记录硬件/软件版本；无法保证完全确定性时标明。
8. 使用`ruff`（格式和静态检查）、`pytest`，可选`mypy`；不可通过删除断言或一律skip来让CI变绿。
9. 参数和代码分开；指标统一由评测模块计算，避免每个实验自行重定义AP/Recall。
10. 单独关注效率：只对需要梯度的分支调用反传；注意多卡进程间重复下载和CPU RAM开销。

## 7. 最低测试矩阵

| 测试层 | 例子 | 必须性质 |
|---|---|---|
| 无GPU单测 | 合成PNG的读取/尺寸、16位保持、非法文件处理 | 不需要任何模型下载 |
| schema单测 | 空objects、合法多类、越界bbox、未知metadata | 错误输入有清晰报错 |
| tile单测 | 正常tile、右下边界、不整除尺寸、overlap、padding | 100%覆盖；offset回填正确 |
| 坐标单测 | XYXY↔xywh/COCO、tile→global、归一化1000 round-trip | 最大容许误差显式定义、不能重复归一化 |
| 渲染单测 | 画红框保存PNG，与原图像素逐点比较 | **只改框线像素，宽高不变** |
| SFT格式单测 | 已审核的图像+messages，模型processor可读 | 无占位图片路径残留 |
| GPU smoke | 真实权重+一张图进行一次forward或generate | 实际完成，非mock，记录峰值显存 |
| 训练配置检查 | YAML可解析；参数在当前swift版本受支持 | 失败时明确提示，不得谎报训练成功 |
| 分布式短跑 | 4GPU若有权限，少量step启动、日志聚合 | rank一致、不挂死、不越界显存 |
| 评测回归 | 固定微型真值集 | 修改坐标/NMS/评分代码时数值可追踪 |

测试优先使用小型合成**工程**fixture检查坐标、接口和程序正确性；不能拿这些数据推断SAR模型识别能力。真实科研结果必须来自独立、经核验的测试集。

## 8. Codex自主工作和汇报格式

每个任务先输出一个不超过10行的实施计划，说明改动文件、依赖、验证命令、风险；然后最小可行改动→运行测试→汇报结果。若有用户已有代码，先阅读并保持兼容。技术路线变更必须写`docs/ADR-xxxx.md`（新增目录可在首次需要时创建），说明：问题、原决策、改动理由、替代方案、实验/兼容影响，待项目负责人批准后更改**已确定**项。

任务完成报告必须有以下字段：

```text
Task ID: MVP-xx
Status: DONE | PARTIAL | BLOCKED | NOT RUN
Changed files: ...
Commands executed: ...
Tests: passed X / failed Y / not run Z
Artifacts: ...
Actual hardware/software: ...
Measured GPU peak memory (if run): ...
Open risks/next dependency: ...
```

- **DONE**：完成对应任务所有验收条目；**PARTIAL**：仅子集完成；**BLOCKED**：外部资源/权限阻塞；**NOT RUN**：尚未执行。
- 每个PR只解决一个明确任务或紧密依赖任务；禁止为赶工一口气提交多个未验证模型分支。
- 能力尚未实现时，用`NotImplementedError`或显式`feature_enabled=false`，不应输出看似合理的伪检测结果。
- 不为提高表观成绩而从测试集反复挑样、修改阈值并回填训练，必须记录验证集选择阈值的流程。

## 9. 科研实验和安全边界

- 对所有实验留存可复现配置及原始日志；论文叙述区分“设计意图”“实验观察”“可支持的结论”。
- 数据集、权重、仿真模型须遵守许可与敏感数据处理要求；未知版权/访问条款不得自动上传公开仓库。
- 若示例包含机场与舰船细粒度类别，仅作图像识别科研任务；不要扩展为未要求的人员跟踪、秘密信息收集或现实决策自动化。
- 模型拒识与可靠性应量化风险；禁止宣称识别结果可100%无需核验。

## 10. 开发前必读参考

- `PROJECT_CONTEXT.md`：研究目标、模块、数据和实验定义。
- `TASKS.md`：严格实施顺序、P0起始MVP和验收门槛。
- Qwen3-VL官方：https://github.com/QwenLM/Qwen3-VL
- 官方ms-swift实践：https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Qwen3-VL-Best-Practice.md
- 官方Grounding说明包含在上述文档的Training章节；当前实现需再次对照所安装版本。

> **给Codex的第一条行动要求**：不要先写完整Scene/Physical/Reasoning框架。先执行`TASKS.md`的`MVP-01`至`MVP-06`，形成可检验的最小工程，再逐阶段扩展。
