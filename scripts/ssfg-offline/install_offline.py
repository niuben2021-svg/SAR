#!/usr/bin/env python3
"""Install only missing Python packages from local wheels. Never access PyPI."""
import sys,pathlib,json,platform,importlib,importlib.metadata,subprocess,hashlib
root=pathlib.Path(__file__).resolve().parent
meta_file=root/'target_env.json'
if not meta_file.is_file(): sys.exit('Missing target_env.json')
info=json.loads(meta_file.read_text(encoding='utf8'))
if info['system']!=platform.system() or info['architecture']!=platform.machine():
    sys.exit('Platform mismatch: package targets %s/%s, local %s/%s' % (info['system'],info['architecture'],platform.system(),platform.machine()))
actual=f'{sys.version_info.major}.{sys.version_info.minor}'
if info['python_version']!=actual:
    sys.exit('Python mismatch: package built for %s, current is %s. Use same interpreter/venv as probe_target.py.' % (info['python_version'],actual))
whls=list((root/'wheels').glob('*.whl'))
if not whls: sys.exit('Wheel files are missing! The helper ZIP alone is not the offline bundle. On external network machine run download_on_online.py.')
hash_file=root/'SHA256SUMS.txt'
if not hash_file.exists(): sys.exit('Missing SHA256 manifest')
for line in hash_file.read_text(encoding='utf-8').splitlines():
    hashvalue, rel = line.split(maxsplit=1)
    p=root/rel
    if not p.is_file(): sys.exit('Missing wheel: '+rel)
    if hashlib.sha256(p.read_bytes()).hexdigest()!=hashvalue:
        sys.exit('SHA256 mismatch: '+rel)
print('Wheel SHA256 checksums verified:',len(whls))
mods=[('numpy','numpy'),('scipy','scipy'),('Pillow','PIL'),('h5py','h5py'),('opencv-python-headless','cv2')]
installed_original=info['versions']
try:
    import torch,torchvision
    from torchvision.ops import nms
    assert torch.cuda.is_available(), 'CUDA unavailable'
    print('Torch:',torch.__version__,'TorchVision:',torchvision.__version__,'GPU:',torch.cuda.device_count())
except Exception as exc:
    sys.exit('Existing GPU PyTorch/TorchVision not functioning: '+str(exc))
for conflict in ('opencv-python','opencv-contrib-python','opencv-contrib-python-headless'):
    try: v=importlib.metadata.version(conflict)
    except importlib.metadata.PackageNotFoundError: continue
    sys.exit(f'Conflicting package {conflict}=={v}: do not install multiple OpenCV variants.')
for dist,mod in mods:
    try:
        importlib.import_module(mod)
        print('Keep existing:',dist)
        continue
    except ImportError:
        pass
    except Exception as exc:
        sys.exit(f'Installed module {mod} raises error: {exc}. Fix environment rather than blindly replace its dependencies.')
    name_for_pip=dist
    if dist=='opencv-python-headless': name_for_pip+='==4.10.0.84'
    elif installed_original.get(dist): name_for_pip+='=='+installed_original[dist]
    cmd=[sys.executable,'-m','pip','install','--disable-pip-version-check','--no-index','--no-deps','--find-links',str(root/'wheels'), name_for_pip]
    print('Installing offline:',name_for_pip)
    subprocess.run(cmd,check=True)
    importlib.invalidate_caches()
    # pip-installing imports after ImportError may be affected by Python importlib caches
    importlib.import_module(mod)
print('\nRun final verification')
subprocess.run([sys.executable,str(root/'verify_train_env.py')],check=True)
