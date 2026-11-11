# YOLO11 OBB / Ultralytics 8.4.209 / CUDA 12.1 完整离线镜像

镜像：`sar-yolo11-obb:8.4.209-cu121`。适用 Linux x86_64、RTX4090；沿用先前服务器驱动 535.146.02，宿主须有 NVIDIA Container Toolkit。含 PyTorch2.3.1、TorchVision0.18.1+cu121、CUDA12.1/cuDNN8，以及 YOLO11 n/s/m/l/x 五种官方 OBB 预训练权重，路径为 `/workspace/weights/yolo11n-obb.pt`（其他规模替换 n）。不是已针对 SAR 数据训练的权重。

## 内网导入

下载本 Release 的全部附件，保持文件名，放在同一目录，复制到内网。镜像分卷每份最多1.5GB，必须全部下载；GitHub 自动生成的 Source code ZIP 不是完整镜像。为分卷、合并压缩包和 Docker 展开的镜像层预留几十 GB 空间。

```bash
bash 03_offline_load_and_run.sh
```

脚本校验分卷、合并、校验完整归档、`docker load` 并确认 amd64 架构。

## GPU 检查与使用

```bash
docker run --rm --gpus all --network none --shm-size=8g \
  sar-yolo11-obb:8.4.209-cu121 python /opt/sar-yolo/scripts/check_image.py --gpu

# /data/yolo-data 内需要你的 OBB data.yaml、图片和标注；输出保存在宿主 /data/yolo-runs
docker run --rm --gpus all --network none --shm-size=16g \
  -v /data/yolo-data:/data -v /data/yolo-runs:/workspace/runs \
  sar-yolo11-obb:8.4.209-cu121 yolo obb train \
  model=/workspace/weights/yolo11n-obb.pt data=/data/data.yaml \
  device=0,1,2,3 imgsz=1024 batch=4 epochs=100 workers=4
```

推理可使用 `yolo obb predict model=/workspace/weights/yolo11n-obb.pt source=/data/images device=0`，按需挂载输入/输出。数据不打入镜像。YOLO 的 imgsz 会缩放输入；若需保留 SAR 大图原生像素，应自行配置切片、坐标回填与合并，镜像不声称已实现该处理。

## 离线与验证记录

镜像默认 `YOLO_OFFLINE=true`、`YOLO_AUTOINSTALL=false`、关闭遥测 sync。保留官方源码；字体使用 matplotlib 已附的 DejaVu Sans，在 Ultralytics 字体文件名下提供兼容副本，避免联网取 Arial，不能保证中文字体覆盖。内置源码和本发布的源码归档均对应官方 commit `9b57ed7e6fd366a95047aaa0526f542a62b202f1`；Ultralytics 及其模型遵循官方 AGPL-3.0/商业许可条款，LICENSE 已保留。

发布前在无 GPU 的 x86_64 runner 上以 `--network none` 检查依赖、五种预训练模型的 CPU OBB 推理、合成工程样本一次训练/反向/保存，以及导出后重新加载镜像。合成样本不代表 SAR 精度，runner 无法验证 RTX4090、CUDA 算子、四卡训练及实际显存。内网请先执行 GPU 检查，再训练真实数据。实际版本清单、镜像信息、权重 SHA256 和日志随 Release 附带。
