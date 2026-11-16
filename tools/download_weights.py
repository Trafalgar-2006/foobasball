"""Download the pretrained football detectors (YOLOv8x @1280, from github.com/roboflow/sports).

    python tools/download_weights.py
"""
from pathlib import Path

import gdown

WEIGHTS = {  # public Google Drive ids published in roboflow/sports examples/soccer/setup.sh
    "football-player-detection.pt": "17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q",
    "football-ball-detection.pt": "1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V",
}

out = Path(__file__).resolve().parent.parent / "weights"
out.mkdir(exist_ok=True)
for name, file_id in WEIGHTS.items():
    if not (out / name).exists():
        gdown.download(id=file_id, output=str(out / name), resume=True)
