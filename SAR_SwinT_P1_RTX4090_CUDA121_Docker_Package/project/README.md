# SAR Swin-T + 高分辨率 P1–P5 FPN + Faster R-CNN

**本版本已适配你实际的 `dataset/trainval` 与 `dataset/test` 数据结构**：无需移动、重命名或混合数据；官方 `test` 不会用于训练或验证集划分。

## 1. 数据目录（重要）

```text
dataset/
├── trainval/
│   ├── Annotations/    # 四点多边形 TXT，训练用
│   ├── KeyPoints/      # 备用：第一版不读取
│   ├── METAXmls/      # 备用：第一版不读取
│   ├── PNGImages/     # 幅度图 PNG，训练用
│   └── SLCMats/       # 复数 SLC，第二阶段子孔径实验使用
└── test/
    ├── Annotations/
    ├── KeyPoints/
    ├── METAXmls/
    ├── PNGImages/
    └── SLCMats/
```

也支持多包一层目录，例如 `dataset/SL-TRAINVAL/trainval/PNGImages` 与 `dataset/SL-TEST/test/PNGImages`。会递归寻找**父目录严格叫 `trainval` / `test`** 的 `PNGImages`，不接受多个同名候选以免误读。

`PNGImages/xxx.png` 对应 `Annotations/xxx.txt`，必须具有相同文件主名。TXT格式：

```text
x1 y1 x2 y2 x3 y3 x4 y4 类别名 0
```

一行一个目标，末尾的 `0` 目前不解释。使用四点坐标的**水平外接矩形 HBB**进行 Faster R-CNN 训练与评测，暂不使用旋转框 OBB。

### 划分规则

- `trainval`：按文件名对应的原始成像场景分组，默认划分 **85%组训练、15%组验证**；不会把同一组内的不同切片放进训练和验证。
- `test`：全部保留到官方测试集，**不参与再次随机划分**。
- 自动统计所有类别并写入 `split.json`；里面保存相对路径，避免 `trainval` 与 `test` 中同名文件冲突。
- 组规则为去除文件名末尾两个纯数字切片索引，例如 `GF3_..._02048_06759` → `GF3_...`；**需要核对与你真实数据的成像场景划分一致**。如果文件名不同，需调整 `prepare.py` 的 `acquisition()`。
- 若只有一个独立训练成像组，将不创建验证集，避免场景泄漏；如果 `trainval` 和官方 `test` 存在相同的成像标识，会输出警告。

## 2. 训练结构

- 仅使用 PNG 幅度图：单通道重复为3通道以适配 torchvision Swin-T。
- 主干 Swin-T：stride=4/8/16/32 对应 C2–C5；额外从原始输入用轻量卷积生成 **真实 stride=2 的 C1**。
- FPN 融合 P1–P5，每层128通道；Faster R-CNN 检测头。
- RPN基准 anchors：P1/2/3/4/5 = 8/16/32/64/128 px，宽高比0.5/1/2。
- 所有模型共享训练/测试切片尺寸、划分和基本训练条件；可使用 `--no-p1` 做 P1 消融。
- 原始大图（例如1024×1024）使用原生分辨率512×512重叠切片检测，跨切片NMS后汇总。**不缩小整张原图**，避免直升机等小目标像素进一步减少。

## 3. 运行环境

建议Python 3.10+，GPU训练时安装匹配的CUDA PyTorch+TorchVision后再安装 `requirements.txt`。需要 `torchvision` 的C++/CUDA算子可用，且内网提前备好权重。

推荐下载 **TorchVision版** Swin-T ImageNet预训练权重 `swin_t-704ceda3.pth`；不要直接把微软原仓库权重给这个加载器（参数命名可能不同）。

## 4. 数据检查与准备（从工程目录运行）

下面的 `/data/dataset` 是示例，替换为你服务器的真实 `dataset` 目录：

```bash
python analyze_labels.py --data-root /data/dataset

python prepare.py \
  --data-root /data/dataset \
  --val-ratio 0.15 \
  --seed 42 \
  --out split.json
```

正常输出包括 `train / val / test` 各自图片数量以及类别分布。请检查直升机类别是否有训练样本以及小目标尺寸范围。

首次可以用一个有标注的 `trainval` 样本测试前向：

```bash
python smoke_test.py --data-root /data/dataset --device cuda --tile 128
```

该命令仅检查可运行性，不能代表训练精度。

## 5. 先用一张A100正式训练

```bash
python train.py \
  --data-root /data/dataset \
  --split split.json \
  --swin-weights /weights/swin_t-704ceda3.pth \
  --tile 512 --epochs 24 --batch-size 2 --workers 4 \
  --out-dir runs/swin_p1
```

`--batch-size` 按**每张卡**计算；若GPU显存不足改为1。训练会保存 `last.pth`、每5轮一份epoch模型。**当前脚本尚未自动用验证集挑选最佳模型**；正式报告前应在 `val` 上比较各epoch权重，选定后只在官方test上做最终评估。

### 四张A100 DDP

```bash
OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=4 \
  train.py --data-root /data/dataset --split split.json \
  --swin-weights /weights/swin_t-704ceda3.pth \
  --tile 512 --epochs 24 --batch-size 2 --workers 4 \
  --out-dir runs/swin_p1_4gpu
```

## 6. 测试集评估和整图推理

```bash
python evaluate.py \
  --checkpoint runs/swin_p1/last.pth \
  --data-root /data/dataset --split split.json \
  --subset test --overlap 128 --score-thresh 0.05 \
  --out metrics_p1.json
```

输出HBB mAP50、mAP50:95、各类别AP、每张图像平均误检、不同目标面积分组的召回率。`score-thresh`对虚警和召回敏感，方法对比需保持一致；这些是自实现的101点插值指标，正式论文最好再用COCO评测工具交叉核对。

对单张图片进行整图推理：

```bash
python predict.py \
  --checkpoint runs/swin_p1/last.pth \
  --image /data/dataset/test/PNGImages/某样本.png \
  --overlap 128 --score-thresh 0.25 \
  --out prediction.json
```

## 7. P1消融实验

```bash
python train.py --data-root /data/dataset --split split.json \
  --swin-weights /weights/swin_t-704ceda3.pth \
  --tile 512 --epochs 24 --batch-size 2 \
  --no-p1 --out-dir runs/swin_no_p1

python evaluate.py --checkpoint runs/swin_no_p1/last.pth \
  --data-root /data/dataset --split split.json \
  --subset test --out metrics_no_p1.json
```

优先比较**直升机召回率、面积小于256像素目标的召回率、每图虚警数**。P1增加少量参数和特征内存，因此属于工程结构消融，不是严格参数量匹配实验。

## 8. 后续子孔径实验

**第一阶段不读取 SLCMats、METAXmls、KeyPoints**，方便先取得稳定的幅度图检测基线。第二阶段将在保持 `train / val / test` 划分不变的条件下，给检测主干接入复数域子孔径频谱处理分支。必须保证同名PNG/MAT空间配准及子孔径分解方式正确。

## 常见错误

- `Cannot locate trainval/PNGImages`：`--data-root` 应指向共同父目录 `dataset`，而不是 `dataset/trainval`。
- `... lacks a matching TXT`：检查Annotations里对应文件是否同名、TXT是否完整。
- `Old split.json layout`：旧包的split格式不兼容，重新运行本包的 `prepare.py`。
- `CUDA out of memory`：先将 `--batch-size 2` 改成1；P1分支比普通FPN更耗显存。
- `Loaded weights incompatible`：应使用TorchVision发布的Swin-T权重。
- `No validation images`：可能只有一个独立成像组；需要更多独立场景，不要直接从同组随机切片获得看似独立的验证集。
