"""Multi-object tracking (BoT-SORT + camera-motion compensation) and ball track cleaning."""
from bisect import bisect_right

import numpy as np

Box = tuple[float, float, float, float]


def make_tracker(fps: float):
    """Ultralytics BoT-SORT with sparse optical-flow GMC, tuned for panning broadcast cameras.

    The buffer is raised from the default 30 frames (~1 s) to 3 s, so a player who is
    occluded or briefly leaves the frame during a pan gets their old ID back.
    """
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml

    cfg = YAML.load(check_yaml("botsort.yaml"))
    cfg.update(track_high_thresh=0.3, track_low_thresh=0.1, new_track_thresh=0.4,
               track_buffer=int(3 * fps), match_thresh=0.8,
               gmc_method="sparseOptFlow", with_reid=False)
    return BOTSORT(args=IterableSimpleNamespace(**cfg))


def _centre(b: Box) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def fill_ball_gaps(track: dict[int, Box], max_gap: int, max_speed: float,
                   reset_after: int = 3, breaks=()) -> dict[int, Box]:
    """Drop physically impossible jumps, then linearly interpolate short gaps.

    ``max_speed`` is in pixels per frame. After ``reset_after`` consecutive rejections the
    newest detection is trusted (the ball really moved). Nothing is gated or interpolated
    across ``breaks`` (scene cuts).
    """
    breaks = sorted(breaks)

    def crosses(a, b):  # a scene cut c with a < c <= b
        i = bisect_right(breaks, a)
        return i < len(breaks) and breaks[i] <= b

    kept: dict[int, Box] = {}
    last, rejected = None, 0
    for f in sorted(track):
        if last is not None and rejected < reset_after and not crosses(last, f):
            dist = np.linalg.norm(_centre(track[f]) - _centre(kept[last]))
            if dist > max_speed * (f - last):
                rejected += 1
                continue
        kept[f], last, rejected = track[f], f, 0

    out = dict(kept)
    frames = sorted(kept)
    for a, b in zip(frames, frames[1:]):
        if 1 < b - a <= max_gap + 1 and not crosses(a, b):
            ba, bb = np.array(kept[a]), np.array(kept[b])
            for f in range(a + 1, b):
                out[f] = tuple(ba + (bb - ba) * (f - a) / (b - a))
    return out
