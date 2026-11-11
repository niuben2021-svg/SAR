"""Set offline defaults and ship a redistributable DejaVu font fallback."""
import shutil
from pathlib import Path
import matplotlib
from ultralytics import settings
from ultralytics.utils import USER_CONFIG_DIR

settings.update({'sync': False, 'weights_dir': '/workspace/weights',
                 'datasets_dir': '/workspace/datasets', 'runs_dir': '/workspace/runs'})
font = Path(matplotlib.get_data_path()) / 'fonts/ttf/DejaVuSans.ttf'
USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
for alias in ('Arial.ttf', 'Arial.Unicode.ttf'):
    shutil.copy(font, USER_CONFIG_DIR / alias)
for folder in ('datasets', 'runs'):
    Path('/workspace', folder).mkdir(parents=True, exist_ok=True)
print('Offline settings configured; font aliases use DejaVu Sans (not Microsoft Arial; no CJK guarantee)')
