# SAR Swin-T + Rotated Faster R-CNN / MMRotate 0.3.4 (v1)

This is a **Docker build/export toolkit**, NOT a prebuilt Docker image tar.

## Online Linux machine (requires Docker + NVIDIA Container Toolkit for GPU smoke test)

```bash
bash build_and_export.sh
```

This produces `sar-mmrotate-swin-t-rfrcnn-v1.tar` (multi-GB) and its sha256 file.

## Offline internal machine

```bash
export SAR_CLASSES='airplane' # or 'A,B,C' exactly matching labels
bash import_and_run.sh /path/to/sar-mmrotate-swin-t-rfrcnn-v1.tar /path/to/SAR_DATASET
```

Alternative: `docker load -i image.tar`; `docker run --gpus all --ipc=host ...`.

## Data

Expected label format is DOTA 8-point oriented box format, *not* YOLO xywh or 5-number OBB.

```
SAR_DATASET/
  train/images/*.png
  train/labelTxt/*.txt
  val/images/*.png
  val/labelTxt/*.txt
  test/images/*.png
  test/labelTxt/*.txt
```

Each label line: `x1 y1 x2 y2 x3 y3 x4 y4 category 0`, and first two optional DOTA header lines `imagesource:...` and `gsd:...`. Original image shapes, coordinate order, and target labels must be consistent. For single-channel input, use grayscale PNG decoded as 3 channels by OpenCV.

## First iteration configuration

Swin-T, FPN, RotatedFasterRCNN, 1024x1024 resize, 4 GPUs, batch 2/GPU, workers 12/GPU, 40 epochs, AdamW. No early stopping. Initial version uses random Swin-T initialization if `SWIN_CKPT` is not set. For real baseline, mount an official compatible Swin-T pretrained checkpoint at an in-container path and set `SWIN_CKPT` to that path. Exact category names, angle convention, preprocessing and input resizing must be checked against your SAR dataset before accuracy comparison.

If 48 dataloader worker processes cause resource limits, set `WORKERS_PER_GPU=3` (12 total) for a quick fallback.

## Note

The build uses `mmcv-full==1.7.2` binary wheel for cu118/torch2.0/Python3.10 to avoid compilation. Verification has not been run by the assistant because this environment has no Docker daemon. If the smoke test fails, retain `build.log` and `smoke_test.log` for the next iteration.
