# TASKS.md — SAR-VLM 分阶段开发任务与验收规范

> **2026-10-23 用户已确定更新**：首轮底座改为 `Qwen/Qwen3-VL-4B-Instruct`，BF16 LoRA；内网 Docker 20.10.24、麒麟 V10、x86、驱动 595.58.03、2×A100 40GB。下文原 8B/四卡要求作为历史设计参考，当前实现以 `docs/ADR-0002-qwen3vl4b.md` 和 `docs/qwen3vl4b-offline-guide.md` 为准。

> 版本0.1，2026-10-19。目标读者：Codex执行代理/研发成员/项目负责人。  
> 当前项目状态：**P0 CPU 数据/配置/测试链路已实现；Qwen8B 真实加载/前向受当前机器无 CUDA GPU、无权重阻塞，MVP 整体 PARTIAL；没有训练或 SAR 识别评测结果。**
> 当前目标（2026-10-22用户更新）：**麒麟V10、x86、驱动595.58.03、2×A100 40GB**；原四卡方案保留为参考，首个GPU前向可只使用`cuda:0`。基础模型：**Qwen3-VL-8B-Instruct**。
> 状态标签：**【已确定】**架构/需求约束，**【建议尝试】**可调整的实现，**【尚需验证】**研究假设/实测结论。  
> 若仓库已有同等功能，Codex需先审阅再复用/修复，不能机械重复创建。

## 0. 项目总路线及前置约束

**最终研究目标【已确定】**：用户输入SAR全图、查找任务（如“寻找所有飞机/车辆/舰船”）、可选成像元数据/历史阵位/专家知识，输出**原图像素坐标的所有目标候选及其基础/细粒度类别、可定位证据和可信状态**，支持只在原图上叠加红色检测框。

**研发范式【已确定】**：Qwen3-VL-8B开源底座 + 全图场景查询先验 + 原生分辨率多尺度检测头 + SAR物理原型Adapter + 专家竞争假设/证据查验 + 可信分类/拒识；阶段独立训练、按条件有限联合优化。不能仅凭Prompt写规则、不能只做多智能体工具调用、不能把未经校准的预测当可靠真值。

**P0优先于一切高级研发**。在P0完成前，不进入任何需要真实领域性能的SFT/检测训练，更不假设已拥有大规模专业标注或SAR相位数据。

优先级：`P0`=立即/阻塞后续，`P1`=基础模型阶段必须，`P2`=核心研究模块，`P3`=论文级可信与集成。任务状态：`TODO | IN_PROGRESS | DONE | PARTIAL | BLOCKED | NOT_RUN`。本文件初始全为`TODO`；2026-10-21 的实际状态见各 P0 条目与 `reports/runs/mvp_report.md`。

## OFFLINE-01 离线部署准备（2026-10-22）

- **状态**：PARTIAL；CPU 工程与脚本可验证，真实内网 GPU、实际 FAIR-CSAR 样例/标注和容器构建结果分别记录，不能合并为一次成功。
- **数据**：用户已下载 FAIR-CSAR 复图像目标数据集 V1.0；尚未提供本环境可读取的文件，复数编码/变量/标签/许可待实际核对。
- **新增交付**：Dockerfile、完整模型下载与校验、镜像save/load、断网运行脚本、复图像幅度显示副本、双卡LoRA配置、最多10步的显式训练冒烟入口。
- **硬件调整**：`configs/qwen3vl_lora_2xa100.yaml` 每卡batch1/累积8/有效batch16，先单卡再双卡。原 `4xa100` 配置不作为本次启动默认值。
- **实际验证（2026-10-22）**：54项CPU测试通过、1项GPU测试跳过；独立解压源码包复测同结果。小型官方镜像save/load/SHA256流程通过。CUDA基础镜像大层下载重试后停止，完整CUDA镜像构建/真实前向/训练NOT_RUN。
- **文档/证据**：`docs/offline-deployment.md`、`docs/ADR-0001-actual-two-gpu-target.md`、`reports/runs/offline_delivery_report.md`。

## 1. 第一阶段：最小可运行MVP（P0，必须先做）

**MVP严格定义**：在项目环境中，能读取一张测试图像和最小标注/查询记录；能实际加载官方Qwen3-VL-8B模型及processor；对图像+文本执行**一次真实前向推理**；具备可检查的训练配置；能运行最小pytest；输出完整日志。MVP**不要求**已有SAR检测准确率、不要求可信判别、不要求多GPU SFT跑完，更不能把普通文本描述包装成检测结果。

### MVP-00 识别现有仓库与环境（P0）

- **依赖**：无。
- **输入**：仓库根目录、当前服务器shell、GPU访问权限。
- **步骤**：`pwd && find . -maxdepth 2 -type f ...`盘点；记录Python、torch、transformers、ms-swift、CUDA、`nvidia-smi`、`nvidia-smi topo -m`；检查是否已存在模型权重缓存与真实SAR样本路径。检查磁盘余量与联网限制。
- **输出**：`reports/runs/environment.json`、`reports/runs/environment.md`，文件列出实际命令及版本，标记模型/数据有无，Git commit；不得包含Token/私有秘钥。
- **验收**：环境报告字段不缺；`torch.cuda.device_count()`与实际可见GPU一致或说明受限原因；标出框架兼容风险；绝不擅自改系统驱动。
- **状态**：DONE。
- **执行记录（2026-10-21）**：实测 Python 3.12.14、CPU torch，CUDA count=0；nvidia-smi 不存在。配置/默认 Hub 缓存范围内未发现 Qwen8B 权重，未配置真实 SAR 数据。见 reports/runs/environment.json。

### MVP-01 最小工程骨架和配置体系（P0）

- **依赖**：MVP-00。
- **输入**：`PROJECT_CONTEXT.md`和`AGENTS.md`；仓库实际布局。
- **步骤**：建立可安装的`src/sar_vlm`包、`pyproject.toml`、`configs/mvp.yaml`、`tests/`、`.gitignore`、最小README。保存环境/设备/model_path/输入图/输出路径/seed/视觉Token预算等配置；不预建大量空模块。
- **输出**：包安装接口`pip install -e .`；基础配置读取`python -m sar_vlm.cli.check_env --config configs/mvp.yaml`（命令是**待实现的对外契约**）。
- **验收**：本地包可导入；无GPU时配置仍能加载；非法路径/参数有清晰错误；输出写到`outputs/`并处于gitignore。
- **状态**：DONE。
- **执行记录（2026-10-21）**：已创建可安装 src 包、pyproject、YAML、CLI、README；editable 安装和无 GPU 配置检查通过。

### MVP-02 数据读取与Schema验证（P0）

- **依赖**：MVP-01。
- **输入**：一张合法PNG/JPG/TIFF测试图；**一条工程级synthetic fixture**（仅为程序测试，不作为训练真值）；可选经过许可、真实SAR样本。
- **步骤**：实现`SARReader`+`ImageRecord`；记录原图路径、宽高、通道、dtype、图像sha256、基础Metadata及缺失掩码；实现Canonical JSONL schema的读取/校验。原图始终只读；显示归一化仅用于副本。
- **输出**：`src/sar_vlm/data/reader.py`、`schema.py`、`tests/test_reader.py`、`test_schema.py`及`python -m sar_vlm.cli.smoke_data --config configs/mvp.yaml`。
- **验收**：能读取一个测试图并打印尺寸/dtype/hash；合法空目标集支持；错误box、超界、缺字段报错；单测覆盖8bit、可能的16bit样本及坏路径。所有数据来源有`synthetic_fixture/verified/weak/...`标记。
- **状态**：DONE。
- **执行记录（2026-10-21）**：读取/schema/数据冒烟已实现，8-bit 与16-bit PNG/TIFF、非法框和原图只读测试通过；仅使用 synthetic_fixture，真实 SAR 数据仍缺失。

### MVP-03 官方Qwen3-VL-8B模型与processor加载（P0）

- **依赖**：MVP-00、MVP-01。
- **输入**：`Qwen/Qwen3-VL-8B-Instruct`官方模型或经过校验的本地权重路径；一张测试图片。安装版本按官方文档先核对。
- **步骤**：基于`transformers`官方`Qwen3VLForConditionalGeneration`与`AutoProcessor`实现独立adapter；设备优先单卡`cuda:0`、BF16；未兼容FlashAttention时使用默认/SDPA；打印实际加载的模型类、权重路径或revision及显存。
- **输出**：`src/sar_vlm/models/qwen_adapter.py`，模型加载日志，`tests/test_model_smoke.py`（CPU侧仅做导入/mock；GPU的真实测试单独显式标记）。
- **验收**：**实际可加载8B权重**并记录耗时和显存；参数无法加载时保存真实异常、状态`BLOCKED`，不得换成随机小模型却报告“Qwen8B已加载”。
- **状态**：BLOCKED。
- **执行记录（2026-10-21）**：官方模型类可导入、adapter 已实现；无 CUDA GPU/本地权重，尚未实际加载8B权重，不能据导入成功判定 DONE。

### MVP-04 完成一次真实前向推理（P0关键闸门）

- **依赖**：MVP-02、MVP-03。
- **输入**：真实存在的图像路径、自然语言文本查询（例如“请描述这幅SAR图像中可能存在的目标”）、预训练Qwen权重。
- **步骤**：使用正确processor/chat template构建多模态输入，限制视觉Token，调用`model.generate(do_sample=False, max_new_tokens=128)`或官方同等接口；裁除输入token后解码；记录输入形状、图像尺寸、推理耗时、峰值显存、完整文本输出。设置随机种子；`torch.inference_mode()`。
- **输出**：`python -m sar_vlm.cli.smoke_forward --config configs/mvp.yaml --image <path> --query "..."`；`outputs/mvp/forward_output.json`（含`raw_text`、input、模型版本、真实设备）。
- **验收**：**真实预训练Qwen3-VL-8B**完成一次图文前向，不是mock/规则输出；文件可解析；程序exit 0；实际日志可追踪。若文本并未输出可解析检测框，`detections=[]`，不伪造红框。
- **状态**：BLOCKED。
- **执行记录（2026-10-21）**：实际 smoke_forward 返回2，错误为 requested cuda:0, visible CUDA GPUs=0；forward_failure.json 已记录，没有真实生成输出。

### MVP-05 提供训练配置（不要求完成训练）（P0）

- **依赖**：MVP-00、MVP-01；可与MVP-03并行。
- **输入**：官方ms-swift Qwen3-VL最佳实践、4×A100 40GB硬件约束；一份合法SFT小样本JSONL或可信数据路径变量。
- **步骤**：编写`configs/qwen3vl_lora_4xa100.yaml`和`/scripts/train_sft_4xa100.sh`；依据当前安装版本检查`swift sft --help`。初值：`NPROC_PER_NODE=4`、BF16、LoRA rank16/alpha32、每卡batch1、grad_acc4、`max_length=2048`、`IMAGE_MAX_TOKEN_NUM=1024`、梯度检查点、先冻结ViT/aligner、DeepSpeed ZeRO-2可选。训练脚本必须读取路径/参数，不写死私人目录。
- **输出**：可检查的配置、`scripts/train_sft_4xa100.sh --help`/dry-run（若项目定义此入口），`reports/runs/config_validation.json`。
- **验收**：所有字段有值、来源及类型，CLI参数确为安装版本支持；明确`TRAIN_NOT_EXECUTED`直到实际运行；若未准备核验SFT数据不得启动正式训练。**可解析训练配置≠训练已成功。**
- **状态**：PARTIAL。
- **执行记录（2026-10-21）**：LoRA YAML 与仅校验入口已创建。ms-swift 4.6.0 的 swift sft --help 被辅助解析器提前截获，原始结果 INCOMPLETE_HELP；官方 SftArguments 的 Namespace 参数/类型/choices 检查 PASS。未验证数据模板或实际训练启动，TRAIN_NOT_EXECUTED。

### MVP-06 简单测试、原图红框绘制和MVP交付（P0）

- **依赖**：MVP-02、MVP-04、MVP-05。
- **输入**：小型synthetic fixture、独立给定的box坐标、模型前向日志。
- **步骤**：实现`draw_boxes`（只添加红色轮廓，默认3px、不加文字不改灰度；使用新文件输出）、基本bbox检查和Pytest；完成`/scripts/run_mvp.sh`；整理README运行说明和实际失败/阻塞记录。
- **输出**：`outputs/mvp/boxed_fixture.png`、`outputs/mvp/forward_output.json`、`reports/runs/mvp_report.md`、完整pytest日志。
- **验收**：`pytest -q tests/test_reader.py tests/test_schema.py tests/test_draw_boxes.py`通过；框外像素逐点与原图一致；原图hash在运行前后不变；GPU真实前向符合MVP-04；训练配置通过MVP-05检查。若GPU被阻塞，标记`MVP PARTIAL / GPU BLOCKED`，禁止写`DONE`。
- **状态**：PARTIAL。
- **执行记录（2026-10-21）**：CPU 测试和独立给框红框检查通过；run_mvp.sh --cpu 已运行。真实 GPU 前向仍阻塞，完整 MVP 不记 DONE。

### P0一次性交付验收清单

- [x] `python -m sar_vlm.cli.check_env --config configs/mvp.yaml` 可运行并生成真实设备报告。
- [x] `python -m sar_vlm.cli.smoke_data --config configs/mvp.yaml` 实际读取一张图并校验canonical记录。
- [ ] `python -m sar_vlm.cli.smoke_forward --config configs/mvp.yaml --image PATH --query "查找所有飞机"` 完成**真正的8B前向**，并保存原始输出；此时无检测能力也可以成功完成**前向MVP**。
- [ ] `configs/qwen3vl_lora_4xa100.yaml`与`scripts/train_sft_4xa100.sh`可解析、参数存在且经安装版本帮助文本验证。
- [x] 运行`pytest -q`；对坐标合法性与只叠加红框至少有明确通过的自动化断言。
- [x] `README.md`给出一台新服务器的可复制复现步骤、必要环境、运行命令、文件路径、已知限制。
- [x] 报告**真实运行**的时间、设备、内存与失败，不声称实现SAR检测效果。

### MVP入口约定示例（由Codex实现，当前并非已有命令）

```bash
# 0. 探测环境
nvidia-smi
nvidia-smi topo -m
python -m sar_vlm.cli.check_env --config configs/mvp.yaml

# 1. CPU数据/格式冒烟检查
python -m sar_vlm.cli.smoke_data --config configs/mvp.yaml
pytest -q tests/test_reader.py tests/test_schema.py tests/test_draw_boxes.py

# 2. 权重就绪后，单卡真实Qwen前向
CUDA_VISIBLE_DEVICES=0 IMAGE_MAX_TOKEN_NUM=1024 \
python -m sar_vlm.cli.smoke_forward \
  --config configs/mvp.yaml \
  --image /ABS/PATH/TO/VERIFIED_INPUT.png \
  --query "查找所有飞机"

# 3. 验证（不是自动开始）四卡LoRA脚本的配置有效性
swift sft --help
# 经负责人确认数据/资源后才启动scripts/train_sft_4xa100.sh
```

## 2. 第二阶段：数据工程与可评测基础感知（P1，MVP之后）

### DATA-01 批量规范化及来源追踪（P1）

- **依赖**：MVP-02。
- **输入**：允许使用的SARDet-100K、SAR-AIRcraft、SSDD/HRSID、SIVED或项目自有数据及授权/来源清单。
- **任务**：为每一数据源实现独立converter；统一类别词表、metadata缺失标记、bbox、图像hash和许可证；保留`original_source_id`。从原始大图/采集组产生source-group划分，不允许相邻tile跨train/test。
- **输出**：`data/manifests/*.jsonl`、`data/processed/`、数据审计报告`reports/audits/dataset_audit.md`。
- **验收**：随机抽样人工核对转换框/类别；合法框100%；重复图像、近重复及train/test泄漏检查；公开数据源是否真的有相应元数据写明。

### DATA-02 Qwen多模态SFT与Grounding转换（P1）

- **依赖**：DATA-01、MVP-05。
- **输入**：verified的图像/类别/绝对xyxy标注；无目标图像和查询。
- **任务**：生成`messages+images`格式和官方`objects` Grounding格式；明确Qwen内部规范化坐标为0–1000；实现对照的像素框往返转换单测，不让模型输出文本中的相对坐标直接进渲染器。
- **输出**：`data/processed/sft_train.jsonl`、`sft_val.jsonl`、Grounding转换脚本、转换日志。
- **验收**：抽样加载被锁定版本processor/ms-swift模板成功；同一框往返重建误差有约定；空目标样本被保留；无需伪造机型/物理证据标签。

### BASE-01 建立传统SAR目标检测基线 B0（P1）

- **依赖**：DATA-01。
- **输入**：相同train/val/test及类别映射。
- **任务**：用成熟开源检测框架（优先已具备的YOLO/检测代码）训练一个基础多类SAR检测器，支持原图滑窗推理、去重、位置回填，输出可与后续实验一致的ResultJSON。
- **输出**：训练脚本/配置、checkpoint、推理结果、Precision/Recall/mAP及耗时日志。
- **验收**：能在独立验证集计算准确指标；正式训练若未实际发生只报告`NOT_RUN`；保留训练配置及来源。

### BASE-02 Qwen SAR LoRA-SFT基线 B1（P1）

- **依赖**：MVP-04/05、DATA-02。
- **输入**：经核验的SFT/grounding train/val、4×A100。
- **任务**：先100–500条样本短跑，再正式训练；开始冻结视觉主干/aligner，记录峰值显存和Loss；实验后分析是否需要训练视觉对齐器或ViT局部参数。
- **输出**：LoRA checkpoint（已训练时）、完整命令日志、Qwen直接位置生成评测。
- **验收**：短跑能真实反向传播且参数更新；不OOM或记录调整依据；Qwen输出定位经坐标转换后计算独立验证集指标；不以训练Loss下降等同检测成功。

### BASE-03 原图滑窗推理与框合并（P1）

- **依赖**：DATA-01、BASE-01；也服务于BASE-02。
- **输入**：原图、512/1024大小tile（【建议尝试】）、overlap、任一检测后端。
- **任务**：实现tile覆盖、batch推理、坐标回填、边界裁剪、跨tile重叠候选合并（NMS/Soft-NMS或WBF建议尝试）、目标集输出、红框绘制；单元测试包括右下边界/重复目标/空图。
- **输出**：`src/sar_vlm/inference/*`、`tests/test_tiles.py`/`test_coordinates.py`、CLI全图demo。
- **验收**：坐标映射测试正确；原图宽高与框外像素保持；含多个tile的全图推理可成功；记录不同重叠率对漏检和耗时的影响。

### P1结束闸门

- `B0`与`B1`在**同一数据、同一整幅图**上分别保存可比预测；基础AP/Recall可由评测脚本重新计算；全图切片/去重输出稳定；Qwen-SFT训练结果及限制有实际日志。
- 若SFT未达到可用定位质量，**继续采用已验证的Object Head路线**，不可用生成解释掩盖漏检。

## 3. 第三阶段：统一视觉定位与层级搜索（P2）

### DET-01 可学习SAR Object Head与共享视觉特征（P2）

- **依赖**：BASE-03、BASE-02、DATA-01。
- **输入**：Qwen视觉特征或可兼容的共享多尺度特征、目标绝对框及查询embedding。
- **任务**：在视觉分支增加可训练目标头；先比较`Query-conditioned DETR式Head`与`轻量密集检测Head`，首轮选择显存/性能可实测的实现。训练`L_det`（分类+box）；与Qwen的分类/证据分支建立显式ROI token接口。
- **输出**：`models/object_head.py`、检测训练配置、分类/位置损失测试、输出ResultJSON。
- **验收**：反向传播损失有限、梯度有效；标准形状和坐标测试；至少能跑小批量train/val；训练结果和B0/B1比较，不预设增益。

### SCENE-01 场景区域数据及Query Scene Head（P2）

- **依赖**：DATA-01、DET-01。
- **输入**：全图低分辨率特征、自然语言查询、场景区域标签及目标覆盖标签。
- **任务**：训练`A_Q=f(F_global, Query)`区域优先级/搜索热图；实现`knowledge prior`可选输入；先采用软top-k排序+保留兜底网格。
- **输出**：`models/scene_head.py`、`RegionPriority`接口、`L_prior`、搜索区域可视化与诊断日志。
- **验收**：top-k覆盖率可计算；全图目标不被硬先验删除；可以显式禁用Scene Head；与固定网格比对等计算预算的Recall。

### SCENE-02 全图-区域-局部特征交互（P2）

- **依赖**：SCENE-01、DET-01。
- **输入**：全图G、区域R、局部L视觉特征与对应坐标变换。
- **任务**：实现稳定的跨尺度特征融合（【建议尝试】cross-attention/ROI token pooling）；缓存可重用全图信息；控制视觉Token数量，保证高分辨率tile保留信息。
- **输出**：可融合的单一视觉模型训练配置、token预算分析、全图检出结果。
- **验收**：相同视觉预算下基线B2与场景引导B3比较；运行内存真实记录；证明融合/关闭融合的可控性。

### SCENE-03 先验冲突压力测试（P2，H1的必需实验）

- **依赖**：SCENE-02。
- **输入**：机场有/无飞机、机场外飞机、停车场外车辆、历史错误阵位、空场景等verified数据。
- **任务**：对正确先验、错误先验、无先验、随机先验进行统一评测；报告全部目标Recall以及按场景分组的额外漏检。
- **输出**：`reports/experiments/H1_prior_conflict.md`、结构化实验CSV及配置。
- **验收**：能够**证伪**H1；不能只展示成功样例；若场景先验导致漏检增加，修正软权重或扩大兜底预算，并保留原失败日志。

## 4. 第四阶段：历史知识与SAR物理条件原型（P2）

### KNOW-01 历史阵位与专家知识Schema（P2）

- **依赖**：DATA-01、SCENE-01。
- **输入**：历史位置/类别/时间、配准信息、目标结构知识、来源及审核状态。
- **任务**：实现`HistoricalPriorRecord`和`KnowledgeEncoder`；对历史位置转换到当前影像需记录配准精度，无法配准时不激活该先验；编码时间衰减和缺失mask。
- **输出**：知识Schema、结构化测试、先验开关、history-ground truth泄漏检测。
- **验收**：历史阵位已清空时不得直接补框；未来信息不可进入过去时刻预测；history off路径性能可单独评测。

### PHYS-01 SAR元数据和观测支持审计（P2）

- **依赖**：DATA-01。
- **输入**：产品类型、频率/极化、入射角、方位、分辨率、可用SLC与否、传感器来源。
- **任务**：为每个样本建立元数据可用性mask/来源；定义幅度可提取的几何/纹理/阴影/散射强度特征；**缺少SLC时禁止复相位/子孔径分支输出证据**。
- **输出**：`sar_metadata_audit.csv`、`MetadataEncoder`原型、无metadata退化测试。
- **验收**：100%数据有有效性状态；unknown被显式保留；错误/缺失元数据不会silent fail。

### PHYS-02 真实SAR条件原型库1.0（P2）

- **依赖**：PHYS-01、DET-01。
- **输入**：类别确认的SAR ROI、正确成像参数、可能的姿态和结构证据。
- **任务**：构建`PrototypeBank`，按类别/传感器/视角/分辨率等元数据组织真实特征；建立正负匹配对，避免同一目标、相邻tile跨训练测试泄漏。
- **输出**：可检索原型索引、来源表、条件检索API、原型评价集。
- **验收**：检索结果可溯源；无元数据时合理退化；不能把未经验证的电磁仿真图当实测。

### PHYS-03 Physical Adapter与匹配损失（P2，H2主实验）

- **依赖**：PHYS-02、SCENE-02。
- **输入**：目标ROI真实特征、条件物理原型、metadata mask。
- **任务**：训练原型交叉注意力/对比编码器、正负匹配及可观测性预测；把结果投影为物理证据Token，区分`support/conflict/unobservable/unknown`。
- **输出**：`models/physical_adapter.py`、`L_phys`、物理token接口和训练/验证脚本。
- **验收**：无物理/真实原型/有误元数据三组消融；跨入射角、分辨率、传感器的真实集指标；如无足够分组数据，报告`NOT_EVALUABLE`。

### PHYS-04 经过审核的仿真原型（可选，P3，不阻塞主线）

- **依赖**：PHYS-03。
- **输入**：合法使用的目标结构参数、雷达成像元数据、仿真流程说明、真实SAR对齐验证集。
- **任务**：生成有限条件的候选散射/形态模板及质量审计，研究模拟/真实域差；与真实原型对比。
- **输出**：仿真数据manifest、参数记录、域差异实验。
- **验收**：仿真样本与真实样本明确标记，不能以仿真测试结果替代真实泛化效果；若数据/许可/计算不够，标记`DEFERRED`。

## 5. 第五阶段：竞争假设、证据和可信认知（P3）

### EVI-01 专家判别知识与Evidence Grounding数据（P3）

- **依赖**：DATA-01、DET-01、PHYS-01。
- **输入**：具有可信类别标签的高相似目标样本、专家可核验的判别部件/结构/散射证据位置、SAR条件。
- **任务**：建立类别对`K_ij={e1...en}`；证据`type, roi, observability, polarity, provenance, verified`；模型的“看不清”“未观测到”不等于“特征不存在”。
- **输出**：证据标注指南、审核集、标注一致性报告、Grounding训练JSONL。
- **验收**：每条用于Grounding评测的证据都有真实坐标、来源及至少一次审核；未经标注不能凭语言解释伪造真值。

### EVI-02 假设竞争与Evidence Selector（P3，H3主实验）

- **依赖**：EVI-01、PHYS-03或明确关闭物理分支、BASE-02。
- **输入**：候选类别及置信分布、ROI、可观测证据集合。
- **任务**：实现Evidence Gate、Reasoning Query、Hypothesis Tokens；先用监督ROI选择+可观测性分类实现近似信息增益策略；候选集含`unknown`；返回证据引用而非纯长文本。
- **输出**：EvidenceSelector/Reasoner、训练配置和包含证据bbox的分类输出。
- **验收**：`no-evi` vs `text-only` vs `grounded-evi`对比；同一候选框不因长语言解释就被视为正确；报告真实细粒度判别准确率与grounding指标。

### EVI-03 忠实性/反事实干预（P3）

- **依赖**：EVI-02。
- **输入**：真实证据ROI、随机对照ROI及原始SAR切片。
- **任务**：执行遮蔽/合理背景替换/可接受物理干预，记录干预前后预测变化；对照随机区域和非判别区域，分别报告必要性和充分性。
- **输出**：干预执行器、可复现paired evaluation、相关图表数据。
- **验收**：保存干预位置/算法/随机种子；报告不合理干预导致分布漂移的失败案例；不得把任意遮蔽后分数变化等同真实因果证明。

### TRUST-01 可信校准、OOD与拒识（P3，H3/H4主实验）

- **依赖**：EVI-02、独立validation/test、有verified unknown/负样本。
- **输入**：模型logits、证据状态、可选知识冲突状态、真实标签与未知类标记。
- **任务**：实现温度/分组校准建议路线、`accept_fine/accept_coarse/abstain`三级决策，输出risk-coverage、ECE、Brier、OOD AUROC；校准阈值只能在验证集选择。
- **输出**：校准模型、阈值配置、选择性风险曲线、可信标签输出。
- **验收**：评测同时报告覆盖率和错误风险；不能通过全量拒识“实现100%精度”；没校准时输出`calibrated_confidence=null`。

## 6. 第六阶段：统一集成与论文级实验（P3）

### INT-01 统一模型推理API与原图可视化

- **依赖**：SCENE-02、DET-01；其他模块可通过开关缺省关闭。
- **输入**：原图+查询+可选metadata+history+knowledge。
- **任务**：统一`predict(image, query, metadata=None, history=None, knowledge=None, config=...)`，输出ResultJSON；画红框时**原图保持不变**；支持空目标、超大图切片、类别分级、不确定/拒识标识。
- **输出**：推理CLI/API、红框PNG、json、日志；各模块消融开关。
- **验收**：原图hash不变、bbox仅在图内、输出可复用、识别分支关闭仍能检测、物理元数据缺失仍能预测基础类别。

### EXP-01 基线/消融矩阵与科学假设检验

- **依赖**：BASE-01/02、SCENE-03、PHYS-03、EVI-03、TRUST-01（未完成模块标记不可评测）。
- **输入**：相同数据划分/预算、B0~B6配置、真实checkpoint。
- **任务**：生成对照矩阵、按场景和域分组的Recall/AP、基线差异、计算量/速度、先验冲突、物理元数据扰动、证据干预、风险覆盖，至少3个随机种子或解释为何不足。
- **输出**：`reports/experiments/`下的指标CSV/JSON、图表代码、完整日志、失败分析、方法/数据清单。
- **验收**：所有实验结论能追溯config+data+checkpoint+git sha；未运行用`NOT_RUN`而不是空表填数字；效果不如基线也如实记录。

### REL-01 三个月原型与交付包

- **依赖**：INT-01，主要可评测基线达标；高级模块依据完成情况选择性启用。
- **任务**：形成SAR-VLM 0.1/0.2/0.3对应tag、Demo、复现手册、模块性能表、论文实验素材；建立已完成/未完成/不可复现风险总表。
- **输出**：版本化代码、配置、模型权重索引（如有许可证）、评测报告、README、接口demo。
- **验收**：第三方持同等硬件和已获授权的数据可按文档重复运行；任何未实现能力在Demo中明确关闭而不是作假展示。

## 7. 各阶段里程碑和依赖总表

| 阶段 | 任务集合 | 前置条件 | 预期输出 | 完成判定 |
|---|---|---|---|---|
| M0：最小MVP，P0 | MVP-00至06 | 仓库/服务器/模型权重可用 | 读图、Qwen一次真实前向、配置、基础测试 | 前向和测试均真实通过；权重不可用则部分完成/阻塞 |
| M1：基础数据与基线，P1 | DATA-01/02、BASE-01/02/03 | M0 | verified数据、SFT、B0/B1、全图回填 | 同源独立测试集上可重算检测结果 |
| M2：层级搜索，P2 | DET-01、SCENE-01/02/03 | M1 | Query优先级、原生分辨率检测、prior冲突测试 | B2/B3等预算对照可执行 |
| M3：物理与历史，P2 | KNOW-01、PHYS-01/02/03 | M1/M2及可靠metadata | 条件原型、物理证据、历史软先验 | B3/B4/B5可消融且无物理数据退化安全 |
| M4：证据与可信，P3 | EVI-01/02/03、TRUST-01 | M2及经审核证据标签 | 竞争假设、Grounding、拒识 | 真值支持的evidence/risk评估可执行 |
| M5：集成论文，P3 | INT-01、EXP-01、REL-01 | M1起、选择性接入M2-4 | SAR-VLM 0.x及论文实验记录 | 技术结论可追溯、无虚构数字 |

【建议尝试】时间安排：第1月完成M0/M1与部分M2；第2月推进M2/M3；第3月重点M4/M5。设备共享时，可并行标注、数据工程和轻量模块实验，而非始终四卡只跑同一个实验。

## 8. 验收中容易被误判的问题

1. **模型加载成功不等于推理成功**：必须真的有一张图经过视觉处理并完成forward。
2. **前向生成文字不等于目标检测成功**：MVP只报告图文前向；检测准确率等到verified数据+B0/B1之后。
3. **训练配置存在不等于已训练**：检查参数仅是dry-run；训练日志和权重需真实产生。
4. **有红框图不等于存在真实目标**：synthetic fixture证明渲染程序；检测评测必须引用ground truth。
5. **历史高频点位不等于当前目标存在**：历史只能作为软先验；必须验证当前观测。
6. **全图缩放语义不等于小目标可见**：全图用于场景，目标精细定位必须保留原生tile。
7. **有仿真图不等于物理正确**：原型对真实数据的预测增益才是关键。
8. **会讲证据不等于真正依据证据做决定**：用Grounding/干预/负样本评价忠实性。
9. **高接受样本准确率不等于总体可靠**：必须联合报告覆盖率和选择性风险。
10. **4×40GB不等于单卡160GB**：显存策略应以最拥挤的GPU为约束。

## 9. 首个Codex会话的可复制任务指令

将下列文本交给Codex；三个文档放在仓库根目录，无需阅读任何聊天记录：

```text
请先阅读仓库根目录的 PROJECT_CONTEXT.md、AGENTS.md、TASKS.md。
严格执行 TASKS.md 中 MVP-00 到 MVP-06，禁止越级开发高级模块。
首先检查现有代码、硬件和环境；给出最多10行实施计划。
实现最小可运行闭环：测试图像读取与canonical校验、
Qwen3-VL-8B权重/processor真实加载、一次图文前向、
4×A100 40GB LoRA训练配置校验、pytest基础测试、
原图只叠加红色矩形框的渲染测试。
任何真实SAR数据/权重不可得时显式标记BLOCKED，
不得生成假结果或用mock假装通过GPU前向。
输出每个Task ID的状态、修改文件、真实运行命令、测试结果、
环境记录及接下来依赖项。MVP验收前不写Scene Head、Physical
Adapter和复杂联合训练代码。
```

## 10. 通俗版总结

Codex的第一份工作**不是把科研构想一次性写成上万行代码**，而是搭建并证明以下五件事确实可运行：**图像能读、Qwen能加载、图文能前向、训练配置合法、测试能验收**。之后再逐步训练找得全的全图检测器、会选区域的场景网络、能核对SAR散射的物理模块，以及有证据才细分、否则拒识的认知模型。每一步都用真实测试和对照实验来决定是否继续，而不是提前宣布成功。

## OFFLINE-02：4B 双 A100 离线微调交付（2026-10-23）

状态：**PARTIAL**。环境和代码已构建并完成 CPU 工程验证；真实 4B GPU 前向、反传及双卡训练尚未运行。

- [x] 用户授权将首轮模型改为 Qwen3-VL-4B-Instruct，记录 ADR-0002，保留 8B 对照配置。
- [x] 4B/8B 架构校验、4B 下载入口、4B 单卡/双卡 LoRA 配置与启动代码。
- [x] 自定义 CSV → ms-swift JSONL；显式用户审核、分组/相同图像跨 split 泄漏检查。
- [x] 真实前向入口、3 步单卡短跑入口、用户显式启动的全训练集双卡入口；本次未训练。
- [x] 实际构建 CUDA 环境：Python 3.11.17、torch 2.8.0+cu128、torchvision 0.23.0+cu128、
      Transformers 4.57.6、ms-swift 4.6.0、PEFT 0.20.0。完整依赖清单已保存。
- [x] 最终环境断网工程测试：60 passed / 1 GPU deselected；pip check 通过，训练计划解析通过。
      首次容器测试因磁盘不足失败，修复临时目录后重跑通过，原失败日志保留。
- [x] 宿主 CPU 测试：60 passed / 1 GPU skipped；ruff、shell 语法、git diff --check 通过。
- [x] 用户数据准备、权重下载、镜像导入、GPU 检查、短跑、双卡微调、adapter 使用说明。
- [ ] 真实权重下载：本云环境官方模型地址返回 403；用户需外网下载并转入内网。
- [ ] FAIR-CSAR 实际文件格式/变量/标注核对；当前提供显式复图像显示转换和通用标签入口。
- [ ] 麒麟 V10 + Docker 20.10.24 + Container Toolkit + 双 A100 的实际前向/反传/DDP。
- [ ] 训练后的独立测试集准确率、物理信息增益与显存峰值评测；不得填写推测数值。

镜像归档、源码归档及传输校验的最终结果见 `reports/runs/qwen3vl4b_delivery_report.md`。
- [x] 4B 环境镜像实际导出（4,606,175,178 字节），SHA256 校验及 docker load 回导成功。
- [x] 源码与说明独立打包；模型权重与数据集不包含在归档中。

## DELIVERY-02：ARM64 A100 CUDA 12.6 离线交付（2026-10-28）

状态：PARTIAL（正在构建，GPU 训练 NOT RUN）。

- [x] 已读取公开 AMD64 镜像内配置确认 linux/amd64；新增 docker-images/ 下载目录。
- [x] 依据用户截图新增 ARM64/CUDA12.6 Dockerfile、原生 ARM CI、单 A100 配置与启动器。
- [x] 官方不存在指定 2.8.0 ARM64 CUDA12.6 wheel，采用 2.9.1；决策见 ADR-0003。
- [x] 下载、校验、导入、自己的数据准备及微调说明。
- [ ] ARM64 镜像实际构建、离线 CPU 测试与 GitHub Release 下载校验。
- [ ] 目标麒麟 ARM64 + A100 真实 GPU 前向、短跑与训练效果。
