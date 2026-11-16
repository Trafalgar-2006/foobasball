"""Score made-shot detection against the broadcast score bug (free ground truth).

Every score change on the on-screen scoreboard is a made basket: +2/+3 a field goal, +1 a
free throw. Both team scores are read every 0.5 s with the same PARSeq recognizer used for
jersey numbers; only changes that persist count (single-frame OCR glitches don't). Each
detected "made" shot is then matched to a score change within the scorekeeper's lag.
Misses never change the score, so attempts/misses can't be scored this way.

    python tools/eval_shots.py data/basketball_1080p.mp4 outputs/basketball_1080p_summary.json
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sportsvision.jersey import JerseyReader  # noqa: E402

# Score digits of the two teams in the 1080p Junior National Championship score bug.
DEFAULT_BOXES = ["890,975,958,1022", "960,975,1030,1022"]


def score_changes(series: list, persist: int = 2) -> list[tuple[float, int]]:
    """[(time, score or None)] -> [(time of change, points)] for increases seen `persist`x."""
    changes, cur, cand, run, t_cand = [], None, None, 0, None
    for t, v in series:
        if v is None:
            continue
        if cur is None:
            cur = v
        elif v <= cur:  # unchanged, or a misread lower value
            cand, run = None, 0
        else:
            if v == cand:
                run += 1
            else:
                cand, run, t_cand = v, 1, t
            if run >= persist:
                if v - cur <= 3:
                    changes.append((t_cand, v - cur))
                cur, cand, run = v, None, 0
    return changes


def match(shots: list[dict], baskets: list, lag=(-2.0, 8.0)):
    """Greedy one-to-one match of detected makes to score changes in [end+lag0, end+lag1]."""
    used, matched = set(), []
    for s in sorted((s for s in shots if s["outcome"] == "made"), key=lambda s: s["end_time"]):
        for i, (t, _) in enumerate(baskets):
            if i not in used and s["end_time"] + lag[0] <= t <= s["end_time"] + lag[1]:
                used.add(i)
                matched.append((s["end_time"], t))
                break
    return len(matched), matched


def read_scores(video: str, boxes: list, step_s: float) -> list[list]:
    reader = JerseyReader(ROOT / "weights/jersey-number-yolo11n.pt", gpu=False)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    every, f, series = max(1, round(step_s * fps)), 0, [[] for _ in boxes]
    while cap.grab():
        if f % every == 0:
            frame = cap.retrieve()[1]
            for k, (x1, y1, x2, y2) in enumerate(boxes):
                text, conf = reader.recognize(frame[y1:y2, x1:x2])
                series[k].append((round(f / fps, 2), int(text) if text.isdigit() and conf > 0.8 else None))
        f += 1
    return series


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("video")
    p.add_argument("summary", help="<clip>_summary.json written by the pipeline")
    p.add_argument("--boxes", nargs=2, default=DEFAULT_BOXES, help="x1,y1,x2,y2 per team score")
    p.add_argument("--step", type=float, default=0.5, help="seconds between scoreboard reads")
    args = p.parse_args()

    boxes = [tuple(int(v) for v in b.split(",")) for b in args.boxes]
    series = read_scores(args.video, boxes, args.step)
    baskets = sorted(score_changes(series[0]) + score_changes(series[1]))
    shots = json.loads(Path(args.summary).read_text())["shots"]
    made = [s for s in shots if s["outcome"] == "made"]
    field_goals = [b for b in baskets if b[1] >= 2]
    tp_fg, _ = match(shots, field_goals)
    tp_any, matched = match(shots, baskets)
    final = [next((v for _, v in reversed(s) if v is not None), None) for s in series]
    print(json.dumps(dict(
        final_score_read=final, field_goals=len(field_goals),
        free_throws=sum(b[1] == 1 for b in baskets), attempts_detected=len(shots),
        makes_detected=len(made), makes_confirmed_by_scoreboard=tp_any,
        make_precision=round(tp_any / len(made), 3) if made else None,
        field_goal_recall=round(tp_fg / len(field_goals), 3) if field_goals else None,
        matched=matched, baskets=baskets), indent=2))


if __name__ == "__main__":
    main()
