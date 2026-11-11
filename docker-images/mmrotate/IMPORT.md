# SAR Swin-T Rotated Faster R-CNN 完整离线镜像

镜像 `sar-mmrotate:swin-t-rfrcnn-v1`，Linux x86_64 / Python3.10 / CUDA11.8 / PyTorch2.0.1 / TorchVision0.15.2 / MMCV-full1.7.2 / MMDetection2.28.2 / MMRotate0.3.4。

下载 Release 全部附件，放到内网同一目录，执行 `bash load.sh`。它会校验分卷、合并、校验完整归档并导入镜像。请预留几十 GB 空间。GitHub 自动 Source code ZIP 不能用于 docker load。

先检查 GPU（宿主须配置 NVIDIA Container Toolkit）：
```bash
docker run --rm --gpus all --network none --shm-size=8g sar-mmrotate:swin-t-rfrcnn-v1 bash /workspace/scripts/smoke_test.sh
```

四卡训练：
```bash
mkdir -p mmrotate_outputs
docker run --rm --gpus all --network none --shm-size=16g \
  -v /data/SAR_DATASET:/data/sar:ro -v "$PWD/mmrotate_outputs:/workspace/work_dirs" \
  -e SAR_CLASSES=airplane -e BATCH_PER_GPU=2 -e WORKERS_PER_GPU=12 \
  sar-mmrotate:swin-t-rfrcnn-v1 bash /workspace/scripts/train_4gpu.sh
```

数据按附件要求为 `{train,val,test}/{images,labelTxt}`，DOTA 八点坐标标签，每行 `x1 y1 x2 y2 x3 y3 x4 y4 category difficulty`。按实际类别修改 SAR_CLASSES；内存/进程受限时降低 workers。初始配置会将图像 resize 为1024，并非保持原生分辨率切片。

模型沿用附件默认的随机 Swin-T 初始化，不含 SAR 数据或已训练 SAR 权重。若使用预训练权重，需挂载兼容的 MMDetection Swin-T checkpoint 并设置 SWIN_CKPT 为容器路径，不能随意使用 TorchVision 格式。

必要兼容修正：AdamW 配置添加 `_delete_=True`，清除基础 SGD 的 momentum；依赖增加约束，避免 NumPy2 或不兼容的新 OpenCV 被自动引入。其余配置和训练脚本随构建源码附件提供。

发布前在无GPU runner 中验证 pip check、断网配置解析、模型初始化、AdamW 创建、旋转框 CPU IoU/NMS、镜像导出后重新导入。GPU检查、四卡训练、真实数据精度与峰值显存尚未实测；须在内网验收。
