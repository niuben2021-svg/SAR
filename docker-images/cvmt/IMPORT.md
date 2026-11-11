# CVMT CUDA11.8 内网部署操作说明

下载本 Release 全部附件，放入同一目录，执行 `bash load.sh`。脚本校验分卷并合并导入 `cvmt:cu118`；预留几十GB空间。镜像适用 Linux x86_64，Python3.10、PyTorch2.0.1、TorchVision0.15.2、CUDA11.8、MMCV2.0.1 CUDA扩展、MMEngine0.10.5、timm0.9.16。采用匹配的官方 PyTorch CUDA基础环境构建独立镜像，不修改原来的 MMRotate/SSFG 镜像。

## 1. GPU 验证
宿主须配置 NVIDIA Container Toolkit，执行：
```bash
docker run --rm --gpus all --network none cvmt:cu118 python -c 'import torch; assert torch.cuda.is_available(); exec(open("/opt/verify_cvmt.py").read())'
```

## 2. 进入容器
```bash
docker run --rm -it --gpus all --network none --shm-size=12g \
  -v /root/data1:/root/data1 --name cvmt_baseline cvmt:cu118 bash
cd /opt/CVMT
```
代码位于 /opt/CVMT，宿主数据与输出挂载 /root/data1。按实际数据路径修改挂载，输出必须写入挂载目录以便保存。

## 3. 数据、权重及既有补丁
附件提到的 `cvmt_faircsar_A_patch_v2.zip`、FAIR-CSAR 数据与作者 checkpoint 没有上传，本包不包含也不伪造这些文件。你仍需在内网准备对应材料，按附件的转换、patch 与测试命令操作（详见 README_操作说明.md）；先单卡成功，再切换四卡。不要将环境验证成功视为论文精度复现或 checkpoint 推理已通过。

## 4. 验证范围与源码
官方 CVMT 源码固定 commit a75079317972ecffb86a0b2cedfe7258d2dce2e1，镜像与源码归档均含源码和原许可证。新镜像使用作者自己的 MMDetection3.x，清除旧 OpenMMLab1.x 组件；以 /opt/CVMT 优先导入。

发布前检查断网 CVMT/算子导入、pip check、镜像导出再导入。CPU runner 不具备 GPU，CUDA算子实际执行须在内网完成。完整端到端数据/权重测试未运行。实际版本、镜像信息、日志与校验和随附件提供。wheelhouse 已放在镜像 /opt/wheelhouse，内网无需再 pip 下载。
