"""Package wheels with a declared image baseline, never a fabricated live probe."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent
baseline = {
    'python_version': '3.10',
    'python_implementation': 'CPython',
    'architecture': 'x86_64',
    'system': 'Linux',
    'provenance': 'Declared baseline from previously delivered RTX4090 CUDA121 image; NOT a live target probe',
    'versions': {'torch': '2.3.1', 'torchvision': '0.18.1+cu121',
                 'numpy': '1.26.4', 'Pillow': '10.4.0', 'scipy': None, 'h5py': None},
}
(root / 'target_env.json').write_text(json.dumps(baseline, indent=2) + '\n')
shutil.copy(root / 'requirements.lock', root / 'requirements_target.txt')
wheels = sorted((root / 'wheels').glob('*.whl'))
assert len(wheels) == 5, [wheel.name for wheel in wheels]
assert not any(wheel.name.startswith(('torch-', 'torchvision-')) for wheel in wheels)
(root / 'SHA256SUMS.txt').write_text(''.join(
    hashlib.sha256(wheel.read_bytes()).hexdigest() + '  wheels/' + wheel.name + '\n'
    for wheel in wheels))
files = ['target_env.json', 'requirements_target.txt', 'SHA256SUMS.txt',
         'install_offline.py', 'install_offline.sh', 'verify_train_env.py',
         'probe_target.py', 'README_中文.txt', 'INSTALL.md', 'check_cpu.py',
         'offline-tests.log']
with zipfile.ZipFile(root / 'ssfg_full_offline_wheels.zip', 'w', compression=zipfile.ZIP_STORED) as archive:
    for name in files:
        archive.write(root / name, name)
    for wheel in wheels:
        archive.write(wheel, 'wheels/' + wheel.name)
path = root / 'ssfg_full_offline_wheels.zip'
(root / (path.name + '.sha256')).write_text(hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + path.name + '\n')
print('Packaged', len(wheels), 'wheels;', path.stat().st_size, 'bytes')
