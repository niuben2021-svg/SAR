#!/usr/bin/env python3
"""Run in the OFFLINE training interpreter. No network used."""
import json, sys, platform, importlib.metadata, pathlib, os
P = {
  'numpy': 'numpy',
  'scipy': 'scipy',
  'Pillow': 'Pillow',
  'h5py': 'h5py',
  'opencv-python-headless': 'opencv-python-headless',
  'opencv-python': 'opencv-python',
  'opencv-contrib-python': 'opencv-contrib-python',
  'opencv-contrib-python-headless': 'opencv-contrib-python-headless',
  'torch': 'torch',
  'torchvision': 'torchvision',
}
versions = {}
for name, dist in P.items():
    try: versions[name] = importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError: versions[name] = None
info = {
    'python_version': f'{sys.version_info.major}.{sys.version_info.minor}',
    'python_implementation': platform.python_implementation(),
    'architecture': platform.machine(),
    'system': platform.system(),
    'libc': platform.libc_ver(),
    'python_executable': sys.executable,
    'versions': versions,
}
path = pathlib.Path(__file__).resolve().parent / 'target_env.json'
path.write_text(json.dumps(info, indent=2, ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(info, ensure_ascii=False, indent=2))
print('\nSaved:', path)
if info['system'] != 'Linux' or info['architecture'] not in ('x86_64','aarch64'):
    print('WARNING: current downloader supports Linux x86_64/aarch64; platform mismatch.')
