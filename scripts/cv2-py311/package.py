"""Package the exact requested wheels and a verified offline installer."""
import hashlib
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent
wheels = sorted((root / 'wheels').glob('*.whl'))
assert len(wheels) == 2, [p.name for p in wheels]
(root / 'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest() + '  wheels/' + p.name + '\n' for p in wheels))
archive = root / 'cv2_4.11.0.86_numpy1.26.4_py311_linux_x86_64_offline.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as out:
    for file in wheels + [root / name for name in ['requirements.txt', 'install.sh', 'check_cv2.py', 'INSTALL.md', 'SHA256SUMS', 'offline-tests.log']]:
        out.write(file, file.relative_to(root))
Path(str(archive) + '.sha256').write_text(hashlib.sha256(archive.read_bytes()).hexdigest() + '  ' + archive.name + '\n')
print('Complete offline ZIP:', archive.name, archive.stat().st_size, 'bytes')
