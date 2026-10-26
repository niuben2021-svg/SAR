# PROJECT_CONTEXT.md — SAR-VLM 研究上下文与总体技术方案

> **2026-10-23 用户已确定更新**：首轮底座改为 `Qwen/Qwen3-VL-4B-Instruct`，BF16 LoRA；内网 Docker 20.10.24、麒麟 V10、x86、驱动 595.58.03、2×A100 40GB。下文原 8B/四卡要求作为历史设计参考，当前实现以 `docs/ADR-0002-qwen3vl4b.md` 和 `docs/qwen3vl4b-offline-guide.md` 为准。

> 版本：0.1（项目启动规格，2026-10-19）  
> 项目代号：**SAR-VLM**  
> 本文目的：让不掌握历史聊天的 Codex、研究成员和实验负责人能够独立理解**为什么做、做什么、如何训练、如何验证**。  
> 状态标识：**【已确定】**=当前项目的路线/约束决策，**【建议尝试】**=首选实现但允许基于实测调整，**【尚需验证】**=研究假设或实验目标，绝不视为已经取得的结果。

> **2026-10-22部署信息更新（用户报告，尚未远程实测）**：麒麟V10、x86、驱动595.58.03、2×A100 40GB；已下载FAIR-CSAR复图像目标数据集V1.0，实际格式/许可待核对。下文四卡预算属于原方案参考，当前离线交付使用双卡配置，技术路线不变。

## 0. 项目现状与事实边界

- 【已确定】研究对象：**大幅 SAR 影像中按自然语言查询查找全部指定目标，完成位置检测、基础与细粒度识别、可定位证据查验、可信输出/拒识**；典型目标含飞机、车辆、舰船，典型场景为机场/停机坪、道路/停车场、港口/航道。
- 【已确定】预训练底座优先采用 **`Qwen/Qwen3-VL-8B-Instruct`**；采用已有开源底座的 SAR 适配，不从零预训练大模型。
- 【已确定】可用服务器算力为 **4 × NVIDIA A100 40GB**（总显存160GB，**每卡仅40GB，不可等效为一张160GB显卡**）；应验证驱动、CUDA、拓扑、存储及实际可用显存。
- 【已确定】目标是**统一的可训练视觉语言认知模型**，不是由多名外部智能体轮番调用工具、手写规则拼接出的“伪模型”。允许可复现的影像预处理、滑窗切片、坐标回填、缓存和可视化作为工程组件；早期可分模块训练，后期再共享视觉特征与联合优化。
- 【已确定】研究主线：**知识引导的初始感知 → 基于本体约束的后验验证 → 反馈驱动的动态优化**；证据组织采用 `Evidence → Gated Evidence → Reasoning → Hypothesis → Decision`。用词不回退到“先看—后验—反思”。
- 【已确定】项目处于**设计/待实施状态**。本文件**不声称**已实现模型、不声称已经训练、不填写虚构 mAP、精度或GPU占用。此前对话中的目视红框只属于人工/视觉辅助候选标注，不是可靠真值、不是任何模型的评测结果。
- 【建议尝试】三个月完成 SAR-VLM 0.1→0.3 原型与有效消融；这属于研发计划，不是交付保证。

## 1. 研究背景、问题动机

传统 SAR 目标检测通常在小切片上进行固定类别预测。真实应用要求从大幅影像中根据“找到所有飞机/汽车/舰船”等查询**完整发现目标**，并能在多种成像几何、分辨率、背景干扰下解释并核对判别证据。单纯把大图缩小后交给 VLM，会使小目标消失；单纯在全图滑窗穷举，则成本高且缺乏任务条件化的搜索策略；只在 Prompt 中写“飞机应位于机场”容易造成先验偏见和漏检。SAR 特有的视角依赖散射、叠掩、阴影与杂波又使类间判别更加困难。

### 1.1 已凝练的核心科学问题

1. **场景知识如何转化为可学习、又不损失召回的全局—局部搜索机制？** 在大幅 SAR 影像中，如何根据查询分配原生分辨率视觉预算，并在先验错误、历史阵位为空或目标出现于非常规位置时仍检出目标？
2. **成像条件约束的预期散射表征与真实观测怎样形成一致、可查验的证据？** 如何在观测角、频率/极化、分辨率变化下建立目标的条件化散射原型，并解释哪些证据可观测、哪些证据不可观测？
3. **如何基于竞争假设主动发现真正有判别力的局部证据并实现可信决策？** 面对细粒度相似类别，模型能否定位、选择、验证结构/散射证据，并在证据缺失或冲突时降级识别或拒识？
4. **历史与环境先验如何提供帮助而不替代当次观测？** 历史阵位只能改变搜索优先级和假设先验，不能作为当前目标确实存在的证明。

### 1.2 研究目标（均为待通过实验检验的目标）

- O1：【已确定】完成“全图+查询→**目标全集的像素坐标**+基础类别”的可评测闭环，并输出不修改底图内容的红框图。
- O2：【已确定】支持飞机、车辆、舰船等不同查询共享检索框架，按“场景—区域—目标”逐级感知；避免下采样导致小目标被抹去。
- O3：【已确定】输入有效 SAR 元数据后，利用物理散射/几何特征和条件原型支持跨成像条件的识别；**只有幅度图时不能虚构复相位或子孔径证据**。
- O4：【已确定】结合目标结构知识、历史阵位、专家强判别特征，输出可定位的证据、竞争假设和可信状态。
- O5：【尚需验证】与普通 SAR 检测器、Qwen直接Grounding、多尺度无先验模型等基线相比，在目标完整召回、跨条件泛化、细粒度判别及选择性风险上获得可重复的改善。

### 1.3 研究创新命题（论文假设，不是已证实成果）

| ID | 创新命题 | 理论基础/直觉 | 待验证点 |
|---|---|---|---|
| H1 | **目标知识引导的全局—局部协同感知** | 目标—场景关系提供概率先验；多尺度注意力按需投入原生分辨率计算 | 相同计算预算下是否提升 full-image recall，且不恶化先验冲突场景 |
| H2 | **成像条件约束的预期—观测物理证据对齐** | 电磁散射与观察几何决定目标特征的条件分布；实际观测与预期比对可提供检验信号 | 真实/仿真原型能否提升跨传感器、视角、分辨率泛化；仿真域差异影响多大 |
| H3 | **竞争假设驱动的主动判别证据与可信认知** | 信息增益驱动证据选择；证据Grounding及干预测试约束预测对实际证据的依赖 | 证据删除/保留和模型决策是否有可重复的因果相关变化；是否改善选择性风险 |
| H4 | **历史阵位与当前观测的抗冲突融合** | 时空先验可以缩小搜索空间但不能替代当前似然 | 历史错误/过时时漏检、幻觉和校准误差是否受控 |

> 注意：多尺度检测、先验利用、原型匹配、视觉Grounding分别有已有研究基础；**单纯堆叠模块不自动构成新颖性**。论文需通过机制创新、严格对照及消融证明增益来源。

## 2. 技术路线与数据流

### 2.1 总体流程【已确定为设计框架，尚未实现】

```text
输入：SAR幅度/强度影像 I；自然语言查询 Q；可选SAR元数据 M；
      可选历史阵位 H、专家知识 K
                    |
                    v
 A. 知识引导的全局层级感知（Scene Query Head）
    全图语义/任务先验 → 相关区域优先级/Scene tokens
                    |
                    v
 B. 原生分辨率区域感知（Region + Object Head）
    多尺度切片 → 目标候选 boxes → 目标ROI与视觉特征
    * 包含非典型区域的兜底搜索；跨瓦片坐标回填/去重
                    |
                    v
 C. 成像条件约束的SAR证据（Physical Adapter）
    目标结构、散射/阴影/几何、多条件原型 ↔ 真实ROI观测
    → E_part、E_geo、E_phys、grounded evidence
                    |
                    v
 D. Qwen3-VL-8B统一认知分支
    Evidence Gate → Reasoning Query → Hypothesis Tokens
    → 证据选择、假设竞争、可观测性与冲突查验
                    |
                    v
 E. 可信输出
    boxes + coarse/fine class + grounded evidence + uncertainty
    → 接受细类 / 退回粗类 / 拒识
                    |
             [证据不足时可再次获取局部视觉特征]
                    v
 结果JSON + 原图仅叠加红框的PNG + 评测/日志
```

【已确定】内部目标坐标、评测坐标与最终绘制坐标一律使用**原图绝对像素 XYXY**。Qwen3-VL 官方Grounding训练有归一化到0–1000的表示；使用 `ms-swift` 官方 `objects` Grounding数据格式时交由其转换。**不可把训练坐标、模型文本输出和最终像素坐标混为一谈。**

### 2.2 模块接口契约

| 模块/接口 | 输入 | 输出（约定） | 责任/边界 | 状态 |
|---|---|---|---|---|
| `SARReader` | 图像路径、可选metadata路径 | `ImageRecord`、只读原图、宽高、dtype、trace | 不把图像增强结果覆盖原图；MVP先支持PNG/JPG/TIFF可读取模式 | 【已确定】 |
| `TileIndexer` | `ImageRecord`、tile_size/overlap | `TileRecord`（`tile_id`,`origin_xy`,`valid_box`,`transform`） | 覆盖边界区域，保证切片→原图坐标可逆 | 【已确定】 |
| `SceneQueryHead` | 全图特征、查询向量、可选知识 | `RegionPriority`：原图坐标上的区域评分/优先级 | **软引导**而非“机场外直接删除” | 【已确定架构；实现建议尝试】 |
| `ObjectHead` | 原生分辨率tile/区域特征、查询 | `DetectionProposal[]`（像素XYXY、类别、原始分数） | 密集小目标定位不完全依赖LM自由生成坐标 | 【已确定架构；头部形式待比较】 |
| `PhysicalAdapter` | ROI特征、可用的元数据M、物理原型 | `PhysicalEvidence[]`+有效性掩码 | 无物理条件时退化运行，不能编造复相位 | 【已确定架构；实现建议尝试】 |
| `KnowledgeEncoder` | 历史、场景本体、目标判别概念 | 带来源/时间戳的知识embedding和mask | 历史是先验，不是当前检测标签 | 【已确定】 |
| `EvidenceSelector` | ROI、候选类别、可用证据 | evidence ROI、选择权重、可观测状态 | 基于竞争假设选择高判别力证据 | 【已确定架构；策略待比较】 |
| `QwenReasoner` | 视觉、物理、知识、证据tokens | 类别假设、理由结构/证据引用 | 解释必须指向实际ROI，不允许无依据生成 | 【已确定】 |
| `TrustHead` | 分类logits、证据覆盖/冲突、可选元数据 | calibrated score / accept_coarse / accept_fine / abstain | 仅经过校准后才称“可信分数” | 【已确定】 |
| `ResultWriter` | 原图 + 检测结果 | JSON、红框PNG、日志 | **仅更改框线覆盖像素**，保留宽高、原图未框像素 | 【已确定】 |

### 2.3 模块A：场景先验引导的层级搜索

- 理论依据：在目标查找问题中，`P(region | target query, scene)` 可给出分配视觉计算资源的先验；但检测的视觉证据应具有否决先验的能力。
- 【建议尝试】`q=f_text(Q,K)`；`F_G,F_R,F_L=f_vision(I)`；`A_Q=sigmoid(f_prior(F_G,q))`。用可学习区域排序把更多计算预算给机场/港口/道路等相关区域，同时保留固定比例原生分辨率兜底扫描。
- 【建议尝试】三级粒度：G=全局缩略语义；R=场景区域；L=保持采样分辨率的tile/ROI。首版先做确定性滑窗+可学习排序，避免一开始实现复杂的可微裁剪。
- 训练信号：场景区域监督、区域包含目标的监督、目标覆盖率惩罚、查询条件化排名或多标签分类；设置空场景和错误先验负样本。
- 【尚需验证】在等价目标召回/计算预算下是否优于固定网格、硬规则ROI、无先验多尺度模型。
- **严禁**把“飞机→机场”“汽车→停车场”写成不可绕过的筛选器；机场外飞机、道路外汽车、港口外船只均是必测压力用例。

### 2.4 模块B：成像条件约束的散射与结构证据

- SAR条件元数据 `M` 建议含：`center_frequency_hz`/`band`、`incidence_angle_deg`、`look_azimuth_deg`、`range_resolution_m`、`azimuth_resolution_m`、`pixel_spacing_m`、`polarization`、`product_type`、`sensor`、`acquisition_time`、质量/来源字段；目标朝向是**估计量**而不是天然已知真值。
- 【建议尝试】建立 `P(c,M)={P_c,1(M)...P_c,k(M)}` 的**条件化真实SAR原型库**。先按照目标类别、分辨率、相对视角、传感器/极化组织**真实样本**及结构描述；中期再加入电磁仿真样本；连续原型生成器为长期研究内容。
- 【建议尝试】`E_phys = CrossAttention(F_observed, P(c,M))`，生成携带`support/conflict/unobservable/unknown`状态的物理证据token；明确原型来源和元数据缺失掩码。
- 物理边界：**PNG/JPG幅度图可提取形态、亮斑、纹理、投影、阴影；不含可直接使用的真实复相位、原始子孔径、全极化散射矩阵**。仅当确有SLC及相应元数据时，才开发复数、相位或子孔径分支。仿真与真实影像需做跨域验证，不能把仿真像素模板视为真值。
- 【尚需验证】物理原型是提高跨条件识别，还是引入仿真域偏差；做 `no-phys` / `real-prototype` / `sim-prototype` 比较。

### 2.5 模块C：竞争假设与可定位证据查验

- 结构化证据分为 `E_part`（部件）、`E_geo`（几何关系）、`E_phys`（散射/阴影及物理关联）、`E_context`（场景）、`E_history`（历史，仅作先验）。
- 候选假设 `H={H_c1,H_c2,...,H_unknown}`；细粒度示例可包含相似飞机型号，但**只有具有可信类别标签与可观测的判别标注时**才做具体型号监督。
- 【建议尝试】Evidence Gate：`E→Ê`；Reasoning Query：`Ê→R`；Hypothesis Tokens：`R→H`；原型库支持`E/H↔P_c,k`匹配。
- 【建议尝试】用 `argmax_e ExpectedInformationGain(e)/Cost(e)` 作为主动证据选择的概念目标；实际可先用监督的证据ROI/可观测性分类器近似，而非先上强化学习。
- 证据词表建议包含：部件名、相对位置、可观测性、正/负支持关系、成像条件、证据ROI、来源、审核状态。
- 证据忠实性：实验对特征进行遮蔽/替换，对照随机区域；计算必要性/充分性及对应预测变化。SAR相关散射往往耦合，**简单删除强散射像素可能产生不真实图像**，须纳入反事实有效性检查。
- 【尚需验证】是否减少背景捷径和“先猜结论再编理由”的现象；若无Grounding真值，不得报告精确的证据定位指标。

### 2.6 模块D：历史阵位、可信分类及反馈

- 【已确定】历史位置、类别、观测时间、配准精度、变化状态和来源必须共同记录。历史既可引导初始搜索，也可触发漏检复核；**不能推断历史出现=当前存在**。
- 【建议尝试】概念上的概率融合：`P(c | E,K,M) ∝ P(E|c,M) P(c|K)`；工程中注意知识与视觉特征相关造成双重计数，比较late fusion和attention fusion。
- 【已确定】输出三级决策：`accept_fine`（细类证据足够）、`accept_coarse`（只确认大类）、`abstain`（拒识）；必要时返回`absent`或`uncertain_presence`，避免臆造目标。
- 【建议尝试】温度缩放/分组校准、选择性风险阈值、OOD负样本和证据冲突分数。**不能承诺被接受样本“100%准确”**，应报告风险-覆盖曲线和置信区间。
- 【已确定】反馈驱动的动态优化：仅采纳有审核/证据来源的反馈，建立版本化训练数据；不得把模型自说自话的解释作为新真值。

## 3. 总体优化目标与分阶段训练

统一的概念损失（不是第一阶段必须全部实现的训练脚本）：

```text
L = L_task + λ_prior L_prior + λ_phys L_phys
           + λ_evi L_evi + λ_trust L_trust
L_task = λ_det L_det + λ_cls L_cls + λ_sft L_sft
```

| 损失 | 建议构成 | 激活阶段 |
|---|---|---|
| `L_det` | 分类/objectness + L1 box + IoU/GIoU；检测头形式待比选 | 基础定位阶段 |
| `L_cls` | coarse/fine分类、unknown监督 | 分类阶段 |
| `L_sft` | 结构化目标、证据和查询回复的token级监督 | Qwen领域适配起 |
| `L_prior` | 区域查询相关性BCE/ranking + 真实目标覆盖约束 | Scene Head建立后 |
| `L_phys` | 条件原型的对比/匹配/可观测性监督 | 元数据可靠、物理原型具备后 |
| `L_evi` | 证据ROI grounding、证据选择、合理干预一致性 | 证据数据可用后 |
| `L_trust` | 校准、拒识/选择性风险、OOD监督 | 独立验证集具备后 |

【已确定】按阶段激活；未实现/无标注的项系数设为0，记录参数和数据版本。**不要凭空设计一大堆不受监督、不可验证的Loss。**

建议训练流程：

1. **S1 SAR基础LoRA-SFT**：Qwen3-VL-8B，先冻结视觉主干和对齐器，训练查询理解、SAR基础类别、目标有无和小规模Grounding。只作为首个适配基线；不声称冻结视觉主干能充分学习SAR视觉特征。
2. **S2 场景层级搜索与Object Head**：保留原生分辨率tile、可学习区域排序，训练目标定位；与直接Qwen Grounding及固定滑窗基线比较。
3. **S3 SAR物理适配**：先真实条件原型对齐，后经验证的仿真原型；成像参数缺失时有受测的退化路径。
4. **S4 假设竞争/证据Grounding/可信拒识**：使用可靠专家标注与硬负样本，最后小学习率联合微调；不保证所有分支三个月都达到可用精度。

### 3.1 建议的4×A100起点（需以实测修正）

| 项 | 启动值【建议尝试】 |
|---|---|
| 底座 | `Qwen/Qwen3-VL-8B-Instruct` |
| 多GPU | `NPROC_PER_NODE=4`，先LoRA + DDP/DeepSpeed ZeRO-2；必要时FSDP/ZeRO-3 |
| 数据类型 | BF16；开启梯度检查点 |
| LoRA | rank 16, alpha 32, `target_modules=all-linear`（以版本适配为准） |
| Batch | 每卡micro batch 1，梯度累积4，effective batch 16（4×1×4） |
| 序列/视觉预算 | `max_length=2048`起，`IMAGE_MAX_TOKEN_NUM=1024`起；逐步profile |
| 主干 | S1初始`freeze_vit=true`, `freeze_aligner=true`；之后依据视觉迁移实验决定解冻范围 |
| 全图视觉 | 粗图概览+原生分辨率512/1024 tile；不能将16K图盲目整体压缩用于小目标精确检测 |

**资源原则**：分别测量峰值显存、每步耗时、吞吐；多卡总显存不代表单卡可装下160GB；多模块初期可在不同卡上并行实验，联合训练时切换四卡协同。`nvidia-smi topo -m`决定通信优化策略，不作为算法结果。

## 4. 数据集建设与输入输出规范

### 4.1 数据优先级与来源

- 【建议尝试】基础目标检测：SARDet-100K（聚合多SAR检测数据）、SAR-AIRcraft、SSDD/HRSID（舰船）、SIVED（车辆）及其他审核许可的数据；使用前核验许可证、是否真实SAR/合成切片、已发布split及元数据完备程度。**不要把公开数据集默认当成带完整物理/历史/专家标签的数据集。**
- 【已确定】主张新增层级注释：场景/区域、目标box、类别层级、原图/tile坐标变换、SAR元数据有效性、判别证据位置、历史先验、证据审核状态、先验冲突/OOD标记。
- 【建议尝试】起始数据整理量：全图场景约1000–2000幅、层级样本2000–5000组、物理观测匹配对1000–3000组；仅是**人工工作量规划**，非充分性结论。MVP无需这些数量。
- 【已确定】必须包括真正无目标样本、密集小目标、建筑强散射、目标位于非常规位置、历史阵位有/无目标、低分辨率、跨传感器/视角、未知类；不可只挑容易样本。

### 4.2 统一主数据（Canonical Schema v0.1）

原图坐标：左上原点，`x`向右、`y`向下；所有盒子为`[x1,y1,x2,y2]`，单位像素、右下**半开区间**，要求`0<=x1<x2<=width`、`0<=y1<y2<=height`。不得存储含糊的xywh作为内部真值；第三方转换要标明源格式。

```json
{
  "schema_version": "0.1",
  "scene_id": "illustrative_scene_0001",
  "source_image": "data/raw/illustrative_scene_0001.png",
  "width": 2048,
  "height": 2048,
  "image_type": "SAR_AMPLITUDE",
  "source_group_id": "illustrative_original_acquisition_01",
  "split": "train",
  "query": {"text": "查找所有飞机", "target_classes": ["aircraft"]},
  "metadata": {
    "sensor": null,
    "acquisition_time": null,
    "center_frequency_hz": null,
    "incidence_angle_deg": null,
    "look_azimuth_deg": null,
    "range_resolution_m": null,
    "azimuth_resolution_m": null,
    "pixel_spacing_m": null,
    "polarization": null,
    "product_type": "amplitude",
    "provenance": "unknown"
  },
  "scene_regions": [
    {"region_id": "scene_r1", "label": "airport_apron", "bbox_xyxy": [200, 300, 1700, 1700], "verification": "illustrative_only"}
  ],
  "objects": [
    {
      "object_id": "o1", "bbox_xyxy": [720, 850, 765, 900],
      "coarse_class": "aircraft", "fine_class": null,
      "annotation_status": "illustrative_only",
      "evidence": [
        {"type": "geometry", "name": "possible_wing_structure", "bbox_xyxy": [725, 860, 757, 880], "observability": "unknown", "verification": "pending"}
      ]
    }
  ],
  "historical_priors": [],
  "label_provenance": "illustrative_only"
}
```

**上例纯属格式演示**。不得把这些虚构坐标/标签写入真实训练集。真实数据缺失可使用`null`及有效性掩码，**不能以0替代未知角度**。数据文件含完整图像哈希、版本、采集组和权限/许可证记录。

### 4.3 切片、坐标及模型训练格式

- `tile_bbox_original=[ox,oy,ox+w,oy+h]`；局部框`b_local=[xl1,yl1,xl2,yl2]`回填为`b_global=[xl1+ox,yl1+oy,xl2+ox,yl2+oy]`，考虑padding后有效区域裁剪；原图越界必须报错或显式clip并记录。
- 若图像确实缩放，必须保存独立的仿射变换/逆变换；不要用只适用于无缩放tile的平移公式。
- 独立 `ResultJSON` 统一原图绝对坐标；可选 `bbox_norm1000` **只用于Qwen的Grounding训练/解析边界**。转换公式采用原图宽高，并且只转换一次。
- Qwen `ms-swift` 数据可以使用 `messages`+`images` 的多模态JSONL；若使用官方 Grounding `objects` 模板，`objects.bbox`保持**绝对坐标**，由ms-swift模板转换为Qwen所需相对格式；见下文官方文档。所有转换需有单元测试和真实样例round-trip验证。

一个最小**通用SFT**训练行（非 Grounding模板，回答只用于训练数据管线示范）：

```json
{"messages":[{"role":"user","content":"<image>这是一幅SAR图像吗？请判断是否存在可明确识别的飞机。"},{"role":"assistant","content":"无法依据此示例断定具体目标，需图像证据核验。"}],"images":["/ABS/PATH/TO/VERIFIED_IMAGE.png"]}
```

不能把示例答案对所有图片一概套用。纯结构化坐标输出格式应在任务数据层另行映射为Grounding监督，不混用 `bbox_norm1000` 和像素坐标。

### 4.4 预测结果契约（建议v0.1）

```json
{
  "schema_version": "0.1",
  "scene_id": "actual_scene_id",
  "query": "查找所有飞机",
  "image_size": [2048, 2048],
  "coordinate_system": "original_pixel_xyxy_half_open",
  "detections": [
    {
      "id": "p1", "bbox_xyxy": [100, 200, 140, 245],
      "coarse_class": "aircraft", "fine_class": null,
      "raw_model_score": null, "calibrated_confidence": null,
      "decision": "unverified_candidate",
      "evidence": [], "model_version": "untrained_or_tbd"
    }
  ],
  "run_metadata": {"weights": null, "config_hash": null, "timestamp": null}
}
```

【已确定】在校准前`calibrated_confidence=null`，不要将模型口头百分比伪装成经校准概率。MVP仅需`raw_text`/模型前向输出与可验证的输入trace；模型尚未具备检测能力时可以`detections=[]`，**禁止伪造预测框来通过测试**。

### 4.5 数据划分与质量控制

- 根据`source_group_id`（原始整幅影像/采集批次/地域/目标身份按研究目的确定）进行group split，**不得将同一幅原图的相邻tile随机分给train和test**。
- 物理和历史实验额外执行传感器留出、时间留出、地域留出；不能让未来历史记录流入过去时点预测。
- 数据清单需区分：人工确认真值、文献真值、弱标注、模型候选、示例。弱标注不得混入核心测试真值。
- 真实SAR与合成SAR、幅度图与复数数据必须标记；各自报告性能，避免类别/域分布泄漏。

## 5. 实验与评测设计（全部尚未运行）

### 5.1 基线矩阵

| ID | 比较对象 | 要证明/排除的因素 |
|---|---|---|
| B0 | 经典/主流SAR检测器（例如YOLO系或Deformable DETR系） | 普通密集检测上限/成本 |
| B1 | 原始Qwen3-VL-8B直接对全图Grounding | 纯VLM定位能力及漏检 |
| B2 | Qwen-SFT + 固定多尺度tile/检测头（无查询先验） | 跨尺度本身带来的提升 |
| B3 | B2 + query-conditioned Scene Head | 可学习搜索先验的增量 |
| B4 | B3 + 历史阵位/场景知识 | 历史对召回与误检的影响 |
| B5 | B3/B4 + 条件真实物理原型 | 物理证据有效性 |
| B6 | B5 + Evidence Selector/competition/trust | 细粒度与可信收益 |

**公平性**：固定相同原始大图、train/val/test、类别词表、切片大小、非极大值抑制/去重配置；比较prior效率时还要固定视觉token预算、精细检查tile数或FLOPs/时间预算。

### 5.2 指标

- **目标检出**：按整幅图统计`AP50/AP75/mAP@0.5:0.95`、Precision、Recall、F1、每幅图漏检数、`FPPI`；目标很小时报告尺度分桶和密集度分桶。
- **全图搜索**：`Full-image Recall`、任务相关区域的Target Coverage、Recall-vs-Tiles/Token/GPU-time曲线；不得只用场景分类准确率代替目标召回。
- **细粒度**：coarse/fine分类准确率、混淆矩阵、跨视角/分辨率/传感器性能、易混淆类单列评测。
- **物理证据**：元数据分组、条件原型匹配正确率/检索排名、真实/仿真域差异实验；仅有可靠标注时报告证据Grounding IoU。
- **忠实性与可信**：合理干预下必要性/充分性、随机区域对照、ECE、Brier、Risk-Coverage、Selective Risk、OOD AUROC、接受样本覆盖率。
- **工程性能**：GPU峰值占用、每幅全图耗时、FPS（按明确图像尺寸和是否含预处理统计）、训练吞吐、重复运行一致性。
- 【尚需验证】任何目标值、领先百分比或SOTA主张必须由有时间戳、配置、种子、数据版本的真实实验生成。

### 5.3 必须包含的关键对照

1. 正确先验 vs 过时/错误/缺失先验 vs 随机场景先验。
2. 常规目标位置 vs 非典型位置 vs 空场景；评测prior导致的额外漏检。
3. 幅度-only/no-physical vs 真实物理原型 vs 仿真原型；元数据缺失/错配压力测试。
4. 专家文字描述-only vs 可定位证据 vs 可定位证据+干预查验。
5. 无拒识 vs 校准拒识；同时报告覆盖率，不能通过全部拒识“刷”正确率。
6. VLM纯坐标输出 vs 共享视觉特征上的Object Head，关注密集小目标完整召回。

## 6. 开发与团队路线（建议而非已发生）

| 月份 | 阶段 | 主要交付 |
|---|---|---|
| 第1月 | 基础SFT+全图/切片定位 | SAR-VLM 0.1：查询输入、像素框、可视化、评测闭环 |
| 第2月 | Scene Query、历史先验、条件真实散射原型 | SAR-VLM 0.2：可消融的先验及物理分支 |
| 第3月 | 竞争假设、证据、可信拒识、有限联合训练 | SAR-VLM 0.3研究原型、消融报告及可复现demo |

建议8人：2人Qwen底座训练；2人多尺度检测/搜索；1人物理原型；1人证据与可信；1人数据/评测；1人技术集成与论文。科研负责人应重点负责模块接口及目标可证伪实验。

## 7. 决策记录（防止Codex重新引入已否定方向）

| 决策ID | 决策【已确定】 | 理由 | 不应重新引入的替代 |
|---|---|---|---|
| D-01 | 优先Qwen3-VL-8B-Instruct而非从零训练 | 用开源能力降低初期数据和计算需求 | 自建从零8B预训练作为MVP前置条件 |
| D-02 | 全图低分辨率语义+原生分辨率局部精细感知 | 小目标下采样后会丢失 | 仅将大图缩到1024一次处理即声称检全 |
| D-03 | 场景先验作为软排序而非硬过滤 | 先验可能失效 | 机场外禁止飞机、停车场外禁止汽车 |
| D-04 | 目标定位使用独立可训练Object Head并与Qwen共享视觉表示 | 减少密集小目标自回归坐标遗漏 | 只依赖LM逐个生成所有bbox作为唯一检测路径 |
| D-05 | 物理证据必须由数据支持，缺失条件须mask | 幅度图没有复相位 | 根据PNG虚构SLC/子孔径/极化特征 |
| D-06 | 先真实条件原型，仿真原型后置 | 仿真—真实域差异可能很大 | 一开始要求复杂仿真系统阻塞MVP |
| D-07 | 历史阵位只作先验、必须有时间戳/配准 | 防止错误补检、时间泄漏 | 历史存在就自动增加一个当前目标框 |
| D-08 | 证据必须Grounding且可干预验证 | 防止事后编造解释 | 只训练长篇CoT文本并宣称忠实性 |
| D-09 | 分阶段优化、适度联合微调 | 4×40GB+多分支优化风险 | 第一周全模块端到端/full fine-tune |
| D-10 | 原图绝对XYXY半开坐标作为唯一内部真值 | 统一标注/检测/绘图/评测 | xywh/归一化/像素坐标无标识混用 |
| D-11 | 基于独立真值评测、支持拒识 | 可信不等于绝对正确 | 声称可信集合永远100%正确 |
| D-12 | MVP首先只验证读图→模型加载→单次前向→训练配置→测试 | 降低工程启动阻力 | 将完整论文创新全部作为首个PR验收条件 |

## 8. 官方资料（开发前核对版本）

- Qwen3-VL 官方仓库：https://github.com/QwenLM/Qwen3-VL
- Qwen3-VL模型入口：https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- ms-swift Qwen3-VL最佳实践：https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Qwen3-VL-Best-Practice.md
- ms-swift 参数说明：https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Command-line-parameters.md
- SARDet-100K 官方仓库：https://github.com/zcablii/SARDet_100K

> 框架参数可能随版本调整。Codex应先检查当前服务器依赖与`swift sft --help`，生成锁定版本/环境记录；**不能**把本文“建议命令”视为已经运行通过的命令。

## 9. 通俗版总结

从全图寻找指定目标，先用任务和场景知识决定“哪里值得先看”，再用原生分辨率检查有没有目标；找到候选后，用SAR成像条件预测可能出现的散射/几何证据；当类别相似时，主动去找有判别力的局部证据，结合可靠历史信息验证；证据不足就退回粗类或拒识。最后用独立真值和多组消融证明：**真的找得全、框得准、判得有依据，而不仅是解释得像专家。**
