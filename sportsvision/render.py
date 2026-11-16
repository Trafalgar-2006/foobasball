"""Frame annotation (boxes, IDs, team / jersey labels, HUD) and H.264 writing via ffmpeg."""
import subprocess

import cv2
import numpy as np

from .geometry import EDGES, PITCH, L, W

FONT = cv2.FONT_HERSHEY_SIMPLEX
REFEREE = (0, 215, 255)   # amber (BGR)
BALL = (255, 0, 255)      # magenta: visible on grass and hardwood
RIM = (255, 255, 0)       # cyan
NO_TEAM = (170, 170, 170)


def lab_to_bgr(lab) -> tuple[int, int, int]:
    px = np.uint8([[np.clip(lab, 0, 255)]])
    return tuple(int(v) for v in cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0, 0])


def vivid(bgr) -> tuple[int, int, int]:
    """Kit colour for drawing: same hue, boosted so thin boxes stay visible on grass / wood.

    Measured kit colours are dull (shadow, compression, small crops). White / grey / black
    kits (low saturation) are left as they are.
    """
    h, s, v = (int(x) for x in cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0])
    if s > 40:
        s, v = max(s, 200), max(v, 230)
    return tuple(int(x) for x in cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0])


def _text_color(bg) -> tuple[int, int, int]:
    luma = 0.114 * bg[0] + 0.587 * bg[1] + 0.299 * bg[2]
    return (0, 0, 0) if luma > 140 else (255, 255, 255)


def _tag(img, text, x, y, color):
    (w, h), _ = cv2.getTextSize(text, FONT, 0.5, 1)
    y = max(y, h + 6)
    cv2.rectangle(img, (x, y - h - 6), (x + w + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), FONT, 0.5, _text_color(color), 1, cv2.LINE_AA)


def draw_objects(img, objects: list[dict], team_colors: dict) -> None:
    for o in objects:
        x1, y1, x2, y2 = (int(v) for v in o["bbox"])
        kind = o["class_name"]
        if kind == "ball":
            cx = (x1 + x2) // 2
            cv2.rectangle(img, (x1, y1), (x2, y2), BALL, 2)
            tri = np.array([[cx, y1 - 4], [cx - 8, y1 - 18], [cx + 8, y1 - 18]], np.int32)
            cv2.fillPoly(img, [tri], BALL)
            continue
        if kind == "rim":
            cv2.rectangle(img, (x1, y1), (x2, y2), RIM, 2)
            continue
        if kind == "referee":
            color, text = REFEREE, f"{o['track_id']} REF"
        else:
            team = o["team_id"]
            color = team_colors.get(team, NO_TEAM)
            role = "GK " if kind == "goalkeeper" else ""
            text = f"{o['track_id']} {role}{team or '?'}"
            if o.get("jersey_number"):
                text += f" #{o['jersey_number']}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        _tag(img, text, x1, y1 - 2, color)


def draw_radar(img, objects: list[dict], team_colors: dict, scale: float = 0.03) -> None:
    """Top-down pitch in the bottom-right corner (dugouts / stands on sideline cameras) with
    players, referees and the ball."""
    w, h = int(L * scale), int(W * scale)  # 360 x 210 px
    x0, y0 = img.shape[1] - w - 20, img.shape[0] - h - 20
    roi = img[y0 - 10:y0 + h + 10, x0 - 10:x0 + w + 10]
    roi[:] = (roi * 0.3 + np.array([30, 90, 30]) * 0.7).astype(np.uint8)  # dark-green panel
    for a, b in EDGES:
        (ax, ay), (bx, by) = PITCH[a - 1] * scale, PITCH[b - 1] * scale
        cv2.line(img, (x0 + int(ax), y0 + int(ay)), (x0 + int(bx), y0 + int(by)), (230, 230, 230), 1)
    for o in objects:
        if not o.get("pitch_xy"):
            continue
        px, py = o["pitch_xy"]
        c = (x0 + int(np.clip(px * scale, 0, w)), y0 + int(np.clip(py * scale, 0, h)))
        kind = o["class_name"]
        color = BALL if kind == "ball" else REFEREE if kind == "referee" else team_colors.get(o["team_id"], NO_TEAM)
        cv2.circle(img, c, 4 if kind == "ball" else 6, color, -1, cv2.LINE_AA)
        cv2.circle(img, c, 4 if kind == "ball" else 6, (0, 0, 0), 1, cv2.LINE_AA)


def draw_hud(img, lines: list[tuple[str, tuple | None]], banner: tuple[str, tuple] | None = None):
    """Top-left panel: one row per (text, swatch colour or None); optional big status banner."""
    pad, row = 10, 26
    width = 20 + max(cv2.getTextSize(t, FONT, 0.6, 2)[0][0] for t, _ in lines) + 30
    height = pad * 2 + row * len(lines)
    roi = img[10:10 + height, 10:10 + width]
    roi[:] = (roi * 0.35).astype(np.uint8)  # darken = translucent panel
    for i, (text, swatch) in enumerate(lines):
        y = 10 + pad + row * i + 18
        x = 20
        if swatch is not None:
            cv2.rectangle(img, (x, y - 14), (x + 16, y + 2), swatch, -1)
            x += 26
        cv2.putText(img, text, (x, y), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    if banner:
        text, color = banner
        (w, h), _ = cv2.getTextSize(text, FONT, 1.1, 3)
        x = (img.shape[1] - w) // 2
        cv2.rectangle(img, (x - 14, 14), (x + w + 14, 30 + h), color, -1)
        cv2.putText(img, text, (x, 22 + h), FONT, 1.1, _text_color(color), 3, cv2.LINE_AA)


class VideoWriter:
    """Pipes raw BGR frames into ffmpeg -> browser-playable H.264 MP4."""

    def __init__(self, path, width: int, height: int, fps: float):
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264",
             "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(frame.tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed to encode the annotated video")
