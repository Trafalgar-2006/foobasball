"""Jersey numbers: a YOLO11n locator finds the number box, PARSeq (ONNX, Apache-2.0) reads the
tight crop, and a per-track vote (3+ agreeing reads) decides; then re-ID by number."""
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PARSEQ_REPO = "Felix92/onnxtr-parseq"


def vote_number(reads: list[tuple[str, float]], min_votes: int = 3, min_share: float = 0.5):
    """Confidence-weighted vote over OCR reads of one track; None when not convincing.

    Two-digit bias: a one-digit read contained in a two-digit candidate (a folded shirt or a
    clipped crop turns 32 into "3") counts toward that candidate.
    """
    score, count = defaultdict(float), defaultdict(int)
    for text, conf in reads:
        if text.isdigit() and len(text) <= 2:  # 0, 00 and 08 are legal numbers
            score[text] += conf
            count[text] += 1
    for digit in [t for t in score if len(t) == 1]:
        hosts = [t for t in score if len(t) == 2 and digit in t]
        if hosts:
            host = max(hosts, key=score.get)
            score[host] += score.pop(digit)
            count[host] += count.pop(digit)
    if not score:
        return None
    best = max(score, key=score.get)
    if count[best] >= min_votes and score[best] / sum(score.values()) > min_share:
        return best
    return None


def stitch_by_number(spans: dict) -> dict:
    """Re-identify players across tracklets: same team + same jersey number, never on screen
    together -> same person.

    ``spans``: track_id -> (first_frame, last_frame, team, number). Returns
    {track_id: earlier track_id it continues}. Two tracklets visible at the same time are
    different players (one number is a misread), so an overlap blocks the merge.
    """
    # ponytail: numbers only; appearance was measured useless here (ImageNet features: 20% of
    # teammate pairs closer than half the same-player pairs). Upgrade: sports-trained ReID.
    groups = defaultdict(list)
    for tid, (first, last, team, number) in spans.items():
        if team is not None and number:
            groups[team, number].append((first, last, tid))
    merged = {}
    for members in groups.values():
        members.sort()
        _, head_last, head = members[0]
        for first, last, tid in members[1:]:
            if first > head_last:
                merged[tid] = head
                head_last = last
    return merged


class JerseyReader:
    """Locate the number inside a player crop (YOLO11n), then read it (PARSeq, ONNX).

    A text recognizer needs a tight crop: on whole-torso crops PARSeq hallucinates, on the
    located number box it reads cleanly. No located number = illegible, so no read at all.
    """

    def __init__(self, locator_weights: Path, gpu: bool = True, min_conf: float = 0.6):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        self.locator = YOLO(str(locator_weights))
        self.device = 0 if gpu else "cpu"
        cfg = json.loads(Path(hf_hub_download(PARSEQ_REPO, "config.json")).read_text("utf-8"))
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers() and (gpu or p.startswith("CPU"))]
        self.session = ort.InferenceSession(hf_hub_download(PARSEQ_REPO, "model.onnx"),
                                            providers=providers)
        self.input = self.session.get_inputs()[0].name
        self.vocab = cfg["vocab"]
        self.h, self.w = cfg["input_shape"][1:]
        self.mean, self.std = np.array(cfg["mean"]), np.array(cfg["std"])
        self.min_conf = min_conf

    def _prepare(self, img_bgr: np.ndarray) -> np.ndarray:
        """Aspect-preserving resize into the 32x128 input, zero padded (docTR convention)."""
        scale = min(self.h / img_bgr.shape[0], self.w / img_bgr.shape[1])
        img = cv2.resize(img_bgr, (max(1, round(img_bgr.shape[1] * scale)),
                                   max(1, round(img_bgr.shape[0] * scale))),
                         interpolation=cv2.INTER_CUBIC)
        canvas = np.zeros((self.h, self.w, 3), np.float32)
        y0, x0 = (self.h - img.shape[0]) // 2, (self.w - img.shape[1]) // 2
        canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img[:, :, ::-1] / 255.0
        return ((canvas - self.mean) / self.std).transpose(2, 0, 1)[None].astype(np.float32)

    def read(self, crop_bgr: np.ndarray) -> list[tuple[str, float]]:
        boxes = self.locator.predict(crop_bgr, imgsz=256, conf=0.4, device=self.device,
                                     verbose=False)[0].boxes
        if not len(boxes):
            return []
        x1, y1, x2, y2 = boxes.xyxy[boxes.conf.argmax()].tolist()
        px, py = 0.2 * (x2 - x1), 0.1 * (y2 - y1)  # clipped digits were the main misread
        number = crop_bgr[int(max(0, y1 - py)):int(y2 + py), int(max(0, x1 - px)):int(x2 + px)]
        if number.size == 0:
            return []
        text, conf = self.recognize(number)
        return [(text, conf)] if text and conf >= self.min_conf else []

    def recognize(self, img_bgr: np.ndarray) -> tuple[str, float]:
        logits = self.session.run(None, {self.input: self._prepare(img_bgr)})[0][0]
        probs = np.exp(logits - logits.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        text, confs = "", []
        for i, p in zip(probs.argmax(-1), probs.max(-1)):
            if i >= len(self.vocab):  # end-of-sequence token
                break
            text += self.vocab[i]
            confs.append(p)
        return text, float(np.prod(confs)) if confs else 0.0  # sequence probability
