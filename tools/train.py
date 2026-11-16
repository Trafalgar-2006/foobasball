"""Fine-tune both models on Roboflow basketball-player-detection-3 (CC BY 4.0, 654 NBA frames)
and copy the best weights into weights/: the basketball detector (YOLO11s; 11 classes
collapsed to 6) and the jersey-number locator (YOLO11n on player crops, sport-agnostic).

    python tools/train.py --data-dir datasets                  # both (~25 min on a 4 GB GPU)
    python tools/train.py --data-dir datasets --only numbers
    python tools/train.py --imgsz 1280 --batch 8               # detector at 1280 on a T4
"""
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
HF_REPO = "koppolusameer/basketball-coco-20260416"  # mirror of basketball-player-detection-3 v18
NAMES = ["ball", "ball-in-basket", "player", "player-shooting", "referee", "rim"]
REMAP = {"ball": 0, "ball-in-basket": 1, "player": 2, "player-in-possession": 2,
         "player-shot-block": 2, "player-jump-shot": 3, "player-layup-dunk": 3,
         "referee": 4, "rim": 5}  # "number" and the "basketball" super-category are dropped
PLAYERS = {"player", "player-in-possession", "player-jump-shot", "player-layup-dunk",
           "player-shot-block"}
SPLITS = ("train", "valid", "test")


def _load(src: Path, split: str):
    coco = json.loads((src / split / "_annotations.coco.json").read_text())
    names = {c["id"]: c["name"] for c in coco["categories"]}
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append((names[a["category_id"]], a["bbox"]))
    return coco["images"], anns


def _yolo_line(cls, x, y, w, h, img_w, img_h):
    return f"{cls} {(x + w / 2) / img_w:.6f} {(y + h / 2) / img_h:.6f} {w / img_w:.6f} {h / img_h:.6f}"


def _write_yaml(dst: Path, names: list[str]) -> Path:
    path = dst / "data.yaml"
    path.write_text(f"path: {dst.as_posix()}\ntrain: train/images\nval: valid/images\n"
                    f"test: test/images\nnames: {names}\n")
    return path


def export_detector(src: Path, dst: Path) -> Path:
    for split in SPLITS:
        images, anns = _load(src, split)
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)
        for img in images:
            lines = [_yolo_line(REMAP[n], *box, img["width"], img["height"])
                     for n, box in anns[img["id"]] if n in REMAP]
            shutil.copy(src / split / img["file_name"], dst / split / "images" / img["file_name"])
            (dst / split / "labels" / f"{Path(img['file_name']).stem}.txt").write_text("\n".join(lines))
    return _write_yaml(dst, NAMES)


def export_number_crops(src: Path, dst: Path, pad: float = 0.05) -> Path:
    """One image per player box; label = number boxes lying (>= 80%) inside it."""
    for split in SPLITS:
        images, anns = _load(src, split)
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)
        for img in images:
            frame = cv2.imread(str(src / split / img["file_name"]))
            numbers = [b for n, b in anns[img["id"]] if n == "number"]
            players = [b for n, b in anns[img["id"]] if n in PLAYERS]
            for k, (x, y, w, h) in enumerate(players):
                x1, y1 = max(0.0, x - pad * w), max(0.0, y - pad * h)
                x2, y2 = min(img["width"], x + w * (1 + pad)), min(img["height"], y + h * (1 + pad))
                crop = frame[int(y1):int(y2), int(x1):int(x2)]
                if crop.size == 0:
                    continue
                lines = []
                for nx, ny, nw, nh in numbers:
                    iw = min(nx + nw, x2) - max(nx, x1)
                    ih = min(ny + nh, y2) - max(ny, y1)
                    if iw > 0 and ih > 0 and iw * ih >= 0.8 * nw * nh:
                        lines.append(_yolo_line(0, nx - x1, ny - y1, nw, nh, x2 - x1, y2 - y1))
                stem = f"{Path(img['file_name']).stem}_{k}"
                cv2.imwrite(str(dst / split / "images" / f"{stem}.jpg"), crop)
                (dst / split / "labels" / f"{stem}.txt").write_text("\n".join(lines))
    return _write_yaml(dst, ["number"])


def train(data: Path, model: str, imgsz: int, epochs: int, batch: int, name: str,
          runs: Path, out_name: str) -> None:
    from ultralytics import YOLO

    yolo = YOLO(model)
    yolo.train(data=str(data), imgsz=imgsz, epochs=epochs, batch=batch, patience=15, workers=2,
               project=str(runs), name=name, exist_ok=True)
    yolo.val(data=str(data), split="test", project=str(runs), name=f"{name}-test", exist_ok=True)
    shutil.copy(runs / name / "weights" / "best.pt", ROOT / "weights" / out_name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=ROOT / "datasets")
    p.add_argument("--only", choices=["detector", "numbers"])
    p.add_argument("--imgsz", type=int, default=960, help="detector size (1280 fits a T4)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=4, help="detector batch (8+ on a T4)")
    args = p.parse_args()

    from huggingface_hub import snapshot_download

    src = Path(snapshot_download(HF_REPO, repo_type="dataset",
                                 local_dir=args.data_dir / "bball_coco"))
    runs = (args.data_dir / "runs").resolve()
    (ROOT / "weights").mkdir(exist_ok=True)
    if args.only in (None, "detector"):
        dst = args.data_dir / "bball_yolo"
        data = dst / "data.yaml" if (dst / "data.yaml").exists() else export_detector(src, dst)
        train(data, "yolo11s.pt", args.imgsz, args.epochs, args.batch, "basketball", runs,
              "basketball-yolo11s.pt")
    if args.only in (None, "numbers"):
        dst = args.data_dir / "numbers_yolo"
        data = dst / "data.yaml" if (dst / "data.yaml").exists() else export_number_crops(src, dst)
        train(data, "yolo11n.pt", 256, 40, 64, "numbers", runs, "jersey-number-yolo11n.pt")


if __name__ == "__main__":
    main()
