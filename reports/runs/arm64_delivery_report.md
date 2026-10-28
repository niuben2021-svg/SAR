# DELIVERY-02 ARM64 交付实际结果

Task ID: DELIVERY-02
Status: DONE（镜像、源码、使用说明发布）；目标 GPU 训练 NOT RUN
Changed files: deploy/Dockerfile.arm64; scripts/offline/{build_arm64.sh,run_arm64.sh,check_arm64_args.py,package_arm64.py}; configs/qwen3vl4b_lora_arm64_a100.yaml; .github/workflows/arm64-release.yml; docker-images/; docs/ADR-0003-arm64-cu126.md; TASKS.md
Commands executed: build_arm64.sh; pip check; run_arm64.sh ... cpu; check_arm64_args.py; docker save | gzip | split; GitHub release upload; public download/checksum verification.
Tests: ARM64 container 60 passed / 0 failed / 1 GPU deselected; local CPU 60 passed / 1 deselected; ruff and shell syntax passed; Swift profile fields passed.
Artifacts: https://github.com/niuben2021-svg/SAR/releases/tag/qwen3vl4b-arm64-cu126-20261028
Actual hardware/software: GitHub ubuntu-24.04-arm native aarch64 runner, Python 3.11, torch 2.9.1+cu126, torchvision 0.24.1, transformers 4.57.6, ms-swift 4.6.0, peft 0.20.0. CUDA compiled dependency/import checks passed; no target GPU attached.
Measured GPU peak memory (if run): NOT RUN.
Open risks/next dependency: Kylin V10 ARM64 NVIDIA Container Toolkit, A100 real forward and 3-step smoke, actual dataset/license/labels and complete model weights. No accuracy or GPU memory claims.

Image: linux/arm64, 3,618,078,157 bytes, 3 parts.
SHA256: e9f0b5c459ce7f773396fd8095f3d547b2e1c9a2e17a2c113ef494fa8f80530e
Public verification: 10 small files downloaded and SHA256 verified; 12 asset digests matched SHA256SUMS; all three large part URLs returned HTTP 200 (HEAD). Large parts were not fully re-downloaded, and no local ARM GPU docker load/training was executed.
Workflow: https://github.com/niuben2021-svg/SAR/actions/runs/40696853973 (success).
Main stayed at 4ec0be46347eefd960fc8f20e4926f1c635406ad; no merge. Existing AMD64 release unchanged.
