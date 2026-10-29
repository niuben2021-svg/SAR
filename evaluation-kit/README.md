# SAR 批量推理与精度评估（不训练）

本目录配套完整 SAR-VLM 源码，支持 Qwen3-VL-4B/8B 原始底座、微调后的 PEFT LoRA，以及外部预测文件的离线评分。原来只有 CPU 工程测试和单张模型前向，本版补充独立测试集批量推理与指标计算。无需重建旧 AMD64/ARM64 Docker 镜像；使用本次完整源码目录替换旧源码目录作为挂载即可。

**下载整个配套源码包，不能只下载 evaluation-kit 目录。**

- [完整源码与说明 Release 下载页](https://github.com/niuben2021-svg/SAR/releases/tag/sar-evaluation-20261029)
- [本次独立分支 ZIP 下载](https://github.com/niuben2021-svg/SAR/archive/refs/heads/delivery/evaluation-20261029.zip)
- Release 提供 `sar-vlm-evaluation-source.tar.gz`、`sar-vlm-evaluation-source.zip`、本说明、`SHA256SUMS`、实际环境测试日志。
- 旧 [AMD64 镜像](https://github.com/niuben2021-svg/SAR/releases/tag/qwen3vl4b-offline-20261023) 与 [ARM64 镜像](https://github.com/niuben2021-svg/SAR/releases/tag/qwen3vl4b-arm64-cu126-20261028) 保持原样。镜像、代码、权重、测试数据分别存放，不需要下载另一套多 GB 镜像。

## 1. 先选择你实际有标注的任务

| 任务参数 | 需要的人工审核真值 | 输出指标 |
| --- | --- | --- |
| `--task classification`（默认） | 每张目标切片的 `label` | Accuracy、每类 Precision/Recall/F1、Macro P/R/F1、Weighted-F1、Balanced Accuracy、混淆矩阵、输出覆盖率 |
| `--task detection` | 每张图全部待评类别的水平框与类别，坐标为原图像素 XYXY | TP/FP/FN、Precision/Recall/F1、漏检率、每图误检数、匹配框平均 IoU、AP50、AP75、mAP@0.50:0.95、每类 AP |
| `--task description` | 参考描述 `references`，和/或审核属性 `attributes` | 中文字符 F1、字符级 ROUGE-L F1、规范化文本完全匹配率；属性准确率、属性断言 micro P/R/F1、属性覆盖率、已知属性矛盾率 |
| `--task all` | 上述可用标注 | 同时要求模型输出类别、框、描述和属性；未标注项目明确记为 NOT_ANNOTATED |

FAIR-CSAR 如果只有目标切片和类别，先测分类。**类别标签不能替代框真值；不得把每张切片的整图边界自动当作目标框。** 没有检测框就不能得出检测 mAP；没有描述参考答案也不能得出描述准确率。

本版模型检测采用 Qwen 生成式水平框基线，尚未实现训练后的专门 Object Head 或大图切片搜索。大图缩放可能丢失小目标，不能据此声称“找全目标”。OBB、旋转 IoU、COCO crowd/ignore/面积分组不支持，不能把本版指标标成官方 COCOeval。

## 2. 解压与准备运行路径

```bash
sha256sum -c SHA256SUMS          # 在下载了全部 Release 附件的目录中校验
mkdir -p /srv/code/sar-evaluation
tar -xzf sar-vlm-evaluation-source.tar.gz -C /srv/code/sar-evaluation
cd /srv/code/sar-evaluation/sar-vlm
MODEL=/srv/models/Qwen3-VL-4B-Instruct
DATA=/srv/data/SAR
OUT=/srv/outputs/SAR
mkdir -p "$OUT"

# ARM64 机器：
RUN=scripts/offline/run_arm64.sh
# AMD64/x86_64 机器改为：RUN=scripts/offline/run.sh
bash "$RUN" "$MODEL" "$DATA" "$OUT" evaluate --help
```

权重目录必须是完整的官方 Qwen3-VL-4B/8B 本地模型。代码采用 BF16、SDPA、单 GPU 顺序处理多张图，不是训练。默认使用 `cuda:0`；四张卡也可先只用一张，不需要设置分布式训练参数。需要宿主 NVIDIA 驱动及 Container Toolkit。无权重时可先执行本文第 7 节的纯 CPU 评分演示。

## 3. 直接复用训练时的 CSV

仍使用原训练准备程序的列：

```csv
image,label,source_group_id,split,annotation_status
images/test_001.png,飞机,scene_001,test,verified
images/test_002.png,车辆,scene_002,test,verified
```

例中的文件、类别和审核状态必须换成真实的、经审核的独立测试数据。程序不会替你验证标签真实性。目标切片应按原始场景/目标实例划分来源组，不能只随机打散同一目标不同视角图像。

```bash
# labels.csv 和 images/ 都位于 DATA 下。
bash "$RUN" "$MODEL" "$DATA" "$OUT" evaluate prepare \
  --csv /data/labels.csv --image-root /data \
  --split test --output /outputs/eval-data/test.jsonl
```

转换器会检查 CSV 中不同 split 是否共享 source_group_id 或完全相同的图像内容；也支持只有 test 行的 CSV。已有输出不会被覆盖。可选 CSV 列：`objects_json`、`references_json`、`attributes_json`，内容为下一节相应字段的 JSON（用标准 CSV 工具正确转义）。**原训练 CSV 的 physical_observation 是输入信息，本转换器不会把它冒充参考答案。**

为了检查与已准备训练数据的泄漏，推理时增加：

```text
--exclude-manifest /outputs/sft/data_manifest.json
```

路径应指向原 prepare-sft 生成的真实 `data_manifest.json`，程序只取其中 train 记录对照来源组和图像 SHA256。未提供时报告 NOT_CHECKED，不能声称自动验证了训练/测试独立性。

## 4. 检测与描述使用的 JSONL 格式

每行一张图，以下内容展开只是便于阅读；保存时每条记录要在一行：

```json
{
  "id": "test-001",
  "image": "/data/images/test_001.png",
  "width": 512,
  "height": 512,
  "source_group_id": "scene_001",
  "split": "test",
  "annotation_status": "verified",
  "label": "飞机",
  "objects": [{"label": "飞机", "bbox_xyxy": [120, 140, 300, 320]}],
  "references": ["目标存在若干集中亮散射点，右下侧可见阴影。"],
  "attributes": {"阴影": "明显", "散射分布": "集中"}
}
```

这是**格式示例，数字和文本不是 FAIR-CSAR 的真实标签**，不能直接当成真实测试集。可选 `image_sha256` 校验图像没有被替换。相对图像路径默认相对于 JSONL 文件所在目录；建议容器内统一使用 `/data/...`。

- `label` 表示整张目标切片主要类别；多目标大图可省略它，只评检测。
- `objects: null` 或省略字段：没有检测标注，不进入检测分母。`objects: []`：经审核的负样本，其预测框计为误检。
- 框为原始图像像素 XYXY，右/下边界不包含，要求 `0 ≤ x1 < x2 ≤ width`，不自动裁剪错误框。不支持 crowd/ignore/OBB 标注。
- 多个参考描述放在 `references` 中；未审核描述时省略字段。
- `attributes` 是图像级结构化属性，不是自动从自由文本提取。`null` 表示该属性无法确认/未知，不视为“没有”。多目标逐目标属性绑定尚未实现。
- 软件拒绝未审核、train 或混合 split 的评估清单；synthetic_fixture 只能用于显式工程演示，不进入科研指标。

## 5. 设置类别词表和批量推理

把 `evaluation-kit/protocol.example.json` 复制到 `$DATA/eval/protocol.json`，将类别和属性允许值改为自己的真实标签体系。例子中的“飞机/车辆/舰船”仅示范，不假定为 FAIR-CSAR 的实际完整分类体系。检测分类和细分类应分别使用对应词表，不混算。

```bash
mkdir -p "$DATA/eval"
cp evaluation-kit/protocol.example.json "$DATA/eval/protocol.json"
# 在宿主上编辑 "$DATA/eval/protocol.json"，冻结词表和阈值后再评测试集。

# 测原始底座：不训练、不加载 LoRA。
bash "$RUN" "$MODEL" "$DATA" "$OUT" evaluate infer \
  --manifest /outputs/eval-data/test.jsonl \
  --protocol /data/eval/protocol.json \
  --task classification --output /outputs/eval-base

# 测微调模型：指定 ms-swift 保存的 checkpoint 目录，里面有 adapter_config.json 和 adapter 权重。
bash "$RUN" "$MODEL" "$DATA" "$OUT" evaluate infer \
  --manifest /outputs/eval-data/test.jsonl \
  --protocol /data/eval/protocol.json \
  --adapter /outputs/training/实际训练目录/checkpoints/实际checkpoint目录 \
  --task classification --output /outputs/eval-lora
```

LoRA 必须与所选底座匹配；不要把 adapter 当完整模型权重挂到 MODEL。`--adapter` 路径应处于 `/outputs` 或 `/data` 的已有挂载范围。加载 adapter 使用 `is_trainable=False`，生成使用 `inference_mode()`，程序没有优化器和训练步骤。

有框标注时，把 `--task classification` 改为 `--task detection`；有描述或属性标注时改为 `--task description`；需要统一输出时使用 `--task all`。每次使用新的输出目录。为了公平比较底座和 LoRA，保持测试集、协议、任务、提示词、图像 token 预算和生成长度一致。

参数：`--image-tokens 1024`、`--max-new-tokens 1024`、`--device cuda:0`、`--seed 42`。显存不足可降低 image tokens，但必须在对比实验中一致并记录。长图中目标太多导致生成截断会成为解析失败，不会偷偷丢掉失败样本。

分类提示要求只输出精确类别名称（与此前分类 SFT 的纯类别回答配套），也接受 `{"label":"飞机"}`。检测要求 `{"objects":[{"label":"飞机","bbox_1000":[0,0,100,100],"raw_score":0.8}]}`；description 要求 `{"description":"...","attributes":{"阴影":"明显"}}`；all 组合上述所有字段。模型相对 1000 坐标仅在解析边界转换一次为原图像素，不把归一化坐标当像素。

所有样本使用只含公开类别词表/属性词表的共同提示词。单样本真值、参考描述和属性真值不会进入模型提示。格式错误记录 raw_text 和原因；本版不会根据真值“纠正”模型答案。

## 6. 看结果与复算指标

输出在宿主 `$OUT/eval-base` 等目录：

| 文件 | 内容 |
| --- | --- |
| `report.md` | 可读指标汇总 |
| `metrics.json` | 指标、每类结果、失败率和协议 |
| `confusion_matrix.csv` | 分类混淆矩阵，行是真值、列是预测，含 invalid/abstain 列 |
| `predictions.jsonl` | 每张图的归一化预测、原始生成文本、耗时、图像哈希和推理 trace |
| `comparisons.jsonl` | 每条真值与预测配对，供错误分析；仅推理结束后写出 |
| `prompt.txt` / `protocol.json` | 实际共同提示词及协议快照 |
| `run.json` | 底座/adapter 路径与哈希、代码哈希、软件版本、运行状态；不含训练行为 |

`INCOMPLETE` 表示存在解析/推理失败或缺失预测，程序返回非零状态，同时保留报告。失败的分类/描述样本计零分；检测失败样本保留真值产生漏检。必须同时报告失败率，尤其全背景图像上的推理失败不能仅凭零误检当作成功。

可只用 CPU 对已有预测重新评分，不重新调用模型：

```bash
bash "$RUN" "$MODEL" "$DATA" "$OUT" evaluate score \
  --manifest /outputs/eval-data/test.jsonl --protocol /data/eval/protocol.json \
  --predictions /outputs/eval-base/predictions.jsonl \
  --task classification --output /outputs/eval-rescore
```

外部模型预测同样可用：JSONL 每行至少 `id`，分类字段 `label`，检测字段 `objects:[{label,bbox_xyxy,raw_score}]`（这里已经是原图像素，不是 bbox_1000），描述字段 `description` 与 `attributes`。预测 ID 必须对应真值，重复/额外 ID 被拒绝，缺失 ID 计失败。不要向评分器输入自然语言框坐标。

## 7. 纯 CPU 演示（合成工程样例，不代表 SAR 精度）

```bash
bash "$RUN" "$MODEL" "$DATA" "$OUT" shell -c \
  'python scripts/offline/eval_demo.py --output /outputs/eval-fixture'
bash "$RUN" "$MODEL" "$DATA" "$OUT" evaluate score \
  --manifest /outputs/eval-fixture/samples.jsonl \
  --protocol /outputs/eval-fixture/protocol.json \
  --predictions /outputs/eval-fixture/predictions.jsonl \
  --task all --allow-fixtures --output /outputs/eval-fixture-result
```

演示刻意设置一个错分类和一个负图误检，预期 Accuracy=0.5、AP50=0.5，仅用于确认程序计算正确。未下载模型时 MODEL/DATA 可临时指向存在的空目录，CPU 评分不加载模型。

## 8. 指标定义与文本评价建议

- 分类 Macro 指标只对测试集中有真值支持的类别平均；每类支持数一并输出。零分母按 0 处理。未知/格式错误/拒答进入混淆矩阵最后一列，不能从 Accuracy 分母剔除。由于没有经过校准的分类概率，不计算 AUC/ECE/Brier 或伪造 top-k。
- 检测按类别、按 raw_score 降序、逐图一对一匹配 GT；重复框是 FP。AP 为 101 点插值；AP50/AP75/mAP 的 IoU 阈值分别为 .50/.75/.50:.05:.95；没有 GT 的类别不进入 mAP，但其 FP 仍进入总体 Precision 和每图误检数。
- `score_threshold` 只影响 P/R/F1 的工作点；AP 使用阈值过滤前的分数。每图最多保留 `max_detections` 个最高排序预测，默认100，报告截断数。相同分数按输入稳定排序。缺少任何预测排序分数时 AP 记 null；正阈值遇到无分数预测则记 MISSING_SCORES，不捏造分数。
- **Qwen 输出的 raw_score 是模型自行给出的未校准排序估计，AP 只能作为此生成式检测协议下的探索性指标。** 它不等于传统检测头的可靠概率，不能据此报告置信校准精度。
- 描述文字采用 NFKC、大小写归一化、去标点/空白、Unicode 字符级评分，适用于中英文混合；多参考答案按每项指标取最好匹配再逐样本平均。ROUGE-L 检查顺序，字符 F1 检查重合；本版不是分词版 ROUGE，也不是标准 chrF。
- **散射特性更建议把“文本相似”与“物理/观测正确”分开报告。** 本版的属性准确率/断言 micro F1 衡量已审核字段；已知属性矛盾率=错误明确回答数/全部明确回答数；覆盖率暴露靠大量“无法判断”回避错误的情况。真值为 null 的声明单独计数，无法据此断言是幻觉。
- 可进一步加 **BERTScore（中文语义相似）**、专家盲评的**事实正确性/物理一致性/完整性/证据可定位性**，以及描述与结构化属性间的一致性审核。本版不自动实现这些语义/专家判断；BERTScore 需另备离线编码器，其高分也不证明物理正确。CIDEr/SPICE 需要领域语料或合适语义解析器，不建议直接把通用自然图像的分数当 SAR 物理正确率。
- 对幅度显示图，只评可观测的形态、亮度/散射分布、阴影等；相位/极化/子孔径物理量必须有相应数据和真实标注。本版不会根据幅度图制造这些“真值”。

## 9. 验证范围

20 项新增指标/接口单测覆盖手算 AP、重复框、错误类别、漏检、负样本、无标注、失效输出、中文描述、属性矛盾、坐标转换、CSV 泄漏和不加载模型的评分 CLI；加原工程共 80 项测试。实际平台执行情况以 Release 附带日志和 `reports/runs/evaluation_delivery_report.md` 为准。

目标机器真实 Qwen/LoRA 批量推理和 FAIR-CSAR 精度尚未在云端运行：这里没有用户的真实数据和权重，也没有目标 GPU。软件测试通过不能替代真实模型评估；本次不填写推测准确率、训练效果或 GPU 峰值。
