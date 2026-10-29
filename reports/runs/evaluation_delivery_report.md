# EVAL-01 批量推理与评估交付

Task ID: EVAL-01
Status: DONE (批量推理/评估代码、双架构工程检查、下载发布完成)；真实模型 GPU 推理 NOT RUN
Changed files: src/sar_vlm/evaluation/{schema,metrics,runner,prepare,cli}.py; data/conversion.py; models/qwen_adapter.py; deploy/entrypoint.sh; scripts/offline/{run.sh,eval_demo.py}; tests/test_evaluation.py; evaluation-kit/; .github/workflows/evaluation-release.yml; TASKS.md.
Commands executed: pytest -q -m 'not gpu'; run.sh ... cpu; evaluation.cli score with explicit synthetic fixture; ruff check; bash -n.
Tests: host 80 passed / 1 GPU deselected; original AMD64 image 80 passed / 1 GPU deselected; 20 new metric/contract tests included. Engineered fixture classification accuracy=.5 and AP50=.5 matched intentional synthetic inputs. These are software checks, not SAR accuracy.
Artifacts: independent delivery/evaluation-20261029 branch; release sar-evaluation-20261029 after CI gates.
Actual hardware/software: host Python3.12 CPU; original delivered AMD64 Python3.11, torch2.8.0+cu128/transformers4.57.6/ms-swift4.6.0/peft0.20.0. Native ARM test reuses previously delivered torch2.9.1+cu126 image; consult attached CI log.
Measured GPU peak memory (if run): NOT RUN.
Open risks/next dependency: actual model weights, optional matching LoRA adapter, independent reviewed FAIR-CSAR images and classification/box/text/attribute labels. Full-image resized generation is not a trained Object Head or exhaustive tile detector. Self-reported detection ranking scores are uncalibrated; physical factuality cannot be established by lexical overlap.

No training, driver changes, main merge or fabricated experiment results. No new runtime package is needed. All labels remain outside the inference prompt except the shared public class/attribute vocabularies. Ground-truth/prediction comparisons are written after generation. Missing predictions are included in failure counts and metric denominators.

## 最终发布核验（2026-10-29）

- 原 ARM64 镜像实际加载，并验证原镜像完整归档 SHA256；未重建镜像。
- ARM64 容器 80 passed / 1 GPU deselected，耗时 2.62s；新 evaluate 容器入口及纯 CPU 演示通过。
- GitHub Actions 40810535194 success: https://github.com/niuben2021-svg/SAR/actions/runs/40810535194
- Release: https://github.com/niuben2021-svg/SAR/releases/tag/sar-evaluation-20261029
- 六个附件全部实际公开下载 HTTP200，逐文件 SHA256 与 GitHub digest/SHA256SUMS 一致。
- 完整源码 tar.gz 266624 字节，ZIP 315622 字节；ZIP 完整性检查通过，源码包评估模块与工作区逐字节一致。
- 附件 engineering-fixture-metrics.json 明确 scientific_result=false；其中 .5 等值只是构造输入的预期程序结果。
- main 仍为 4ec0be46347eefd960fc8f20e4926f1c635406ad，未合并。旧 AMD64/ARM64 镜像未替换。
- 源码归档对应构建提交，其 TASKS/报告保留构建前状态；本报告及随 Release 的实际 CI 日志提供最终验证记录。
