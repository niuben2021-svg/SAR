#!/usr/bin/env python3
"""Use on INTERNET-connected host to download Linux binary wheels for OFFLINE server.

The online host Python and OS may differ from the target server. Requires pip >= 21.
"""
import sys,subprocess,pathlib,json,hashlib,zipfile,shutil
ROOT=pathlib.Path(__file__).resolve().parent
info_path=pathlib.Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'target_env.json'
if not info_path.is_file():
    sys.exit('Missing target_env.json: run `python probe_target.py` on the offline server, copy JSON here.')
info=json.loads(info_path.read_text(encoding='utf-8'))
if info.get('system')!='Linux' or info.get('architecture') not in ('x86_64','aarch64'):
    sys.exit('Only Linux x86_64 / aarch64 are supported by this script.')
if info.get('python_implementation')!='CPython':
    sys.exit('Only CPython supported.')
py=info['python_version']
major,minor=map(int,py.split('.'))
if (major,minor) < (3,8) or (major,minor) > (3,12):
    sys.exit('For this frozen OpenCV 4.10 stack, this builder only supports CPython 3.8 through 3.12. Contact us for other versions.')
versions=info['versions']
if not versions.get('torch') or not versions.get('torchvision'):
    sys.exit('Existing PyTorch and TorchVision are required on the OFFLINE target. Do not guess CUDA wheel versions. Need inspect target before downloading.')
if any(versions.get(x) for x in ('opencv-python','opencv-contrib-python','opencv-contrib-python-headless')):
    sys.exit('Target has another OpenCV distribution installed. All use cv2 namespace. Please inspect which one is installed before adding headless OpenCV.')
npv=versions.get('numpy')
if npv and npv.split('.')[0]=='2':
    print('WARNING: Installed NumPy 2.x is detected. OpenCV 4.10 may be compatible but should be tested with this specific environment; do not downgrade NumPy automatically.')
if npv:
    npspec=f'numpy=={npv}'
else:
    npspec=f'numpy=={"1.24.4" if minor==8 else "1.26.4"}'
specs=['opencv-python-headless==4.10.0.84',npspec]
for name in ('scipy','Pillow','h5py'):
    v=versions.get(name)
    specs.append(name+(('=='+v) if v else ''))
wheelhouse=ROOT/'wheels'
wheelhouse.mkdir(exist_ok=True)
arch=info['architecture']
platform_tag='manylinux2014_'+arch
cp=f'cp{major}{minor}'
cmd=[sys.executable,'-m','pip','download','--disable-pip-version-check',
    '--only-binary=:all:','--dest',str(wheelhouse), '--platform',platform_tag,
    '--implementation','cp','--python-version', f'{major}{minor}',
    '--abi',cp,'--abi','abi3','--abi','none',*specs]
print('Downloading offline wheelhouse for:',py,arch)
print('Dependencies:',', '.join(specs))
print('Command:', ' '.join(cmd))
subprocess.run(cmd,check=True)
whls=sorted(wheelhouse.glob('*.whl'))
if len(whls)<5: sys.exit(f'Expected >=5 wheels, got {len(whls)}')
hashes=''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  wheels/'+p.name+'\n' for p in whls)
(ROOT/'SHA256SUMS.txt').write_text(hashes,encoding='utf8')
(ROOT/'requirements_target.txt').write_text('\n'.join(specs)+'\n',encoding='utf8')
archive=ROOT/'ssfg_full_offline_wheels.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
    for path in [ROOT/'install_offline.sh',ROOT/'install_offline.py',ROOT/'verify_train_env.py',ROOT/'README_中文.txt',ROOT/'SHA256SUMS.txt',ROOT/'requirements_target.txt', info_path]:
        z.write(path,arcname=path.name)
    for wheel in whls: z.write(wheel,arcname='wheels/'+wheel.name)
print('SUCCESS',archive,'size MiB',round(archive.stat().st_size/1024**2,1))
