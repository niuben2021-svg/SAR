# EVAL-01 批量推理与评估交付

Task ID: EVAL-01
Status: PARTIAL (代码、宿主与 AMD64 容器工程验证完成；ARM64 CI 与发布待运行；目标 GPU 推理 NOT RUN)
Changed files: src/sar_vlm/evaluation/{schema,metrics,runner,prepare,cli}.py; data/conversion.py; models/qwen_adapter.py; deploy/entrypoint.sh; scripts/offline/{run.sh,eval_demo.py}; tests/test_evaluation.py; evaluation-kit/; .github/workflows/evaluation-release.yml; TASKS.md.
Commands executed: pytest -q -m 'not gpu'; run.sh ... cpu; evaluation.cli score with explicit synthetic fixture; ruff check; bash -n.
Tests: host 80 passed / 1 GPU deselected; original AMD64 image 80 passed / 1 GPU deselected; 20 new metric/contract tests included. Engineered fixture classification accuracy=.5 and AP50=.5 matched intentional synthetic inputs. These are software checks, not SAR accuracy.
Artifacts: independent delivery/evaluation-20261029 branch; release sar-evaluation-20261029 after CI gates.
Actual hardware/software: host Python3.12 CPU; original delivered AMD64 Python3.11, torch2.8.0+cu128/transformers4.57.6/ms-swift4.6.0/peft0.20.0. Native ARM test reuses previously delivered torch2.9.1+cu126 image; consult attached CI log.
Measured GPU peak memory (if run): NOT RUN.
Open risks/next dependency: actual model weights, optional matching LoRA adapter, independent reviewed FAIR-CSAR images and classification/box/text/attribute labels. Full-image resized generation is not a trained Object Head or exhaustive tile detector. Self-reported detection ranking scores are uncalibrated; physical factuality cannot be established by lexical overlap.

No training, driver changes, main merge or fabricated experiment results. No new runtime package is needed. All labels remain outside the inference prompt except the shared public class/attribute vocabularies. Ground-truth/prediction comparisons are written after generation. Missing predictions are included in failure counts and metric denominators.
