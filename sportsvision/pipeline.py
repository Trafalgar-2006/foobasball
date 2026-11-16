"""Broadcast sports analysis: pass 1 detects + tracks and collects evidence (cached), the
whole clip then decides teams / numbers / IDs / shots, and pass 2 renders MP4 + CSV + JSON.
Deciding with the whole clip in hand keeps labels stable instead of flickering per frame."""
import argparse
import csv
import json
import pickle
import time
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .geometry import L, W, fit_homography, to_pitch, warp_box
from .jersey import JerseyReader, stitch_by_number, vote_number
from .render import VideoWriter, draw_hud, draw_objects, draw_radar, lab_to_bgr, vivid
from .shots import ShotDetector
from .teams import (assign_teams, jersey_color, kit_distance, kit_outliers, label_teams,
                    team_switches)
from .tracking import fill_ball_gaps, make_tracker

ROOT = Path(__file__).resolve().parent.parent
TEAM_NAMES = ("A", "B")
PERSON = -1  # class id for COCO person boxes (not in the sport model's names)


@dataclass(frozen=True)
class Sport:
    weights: str
    imgsz: int                       # inference size the weights were trained at
    people: tuple[str, ...]          # detector classes handed to the multi-object tracker
    mask_green: bool                 # mask grass when sampling jersey colour
    ball_weights: str | None = None  # optional dedicated ball detector
    person_weights: str | None = None  # optional COCO person detector merged into people
    pitch_weights: str | None = None   # optional pitch-keypoint model (homography, radar)
    shots: bool = False


SPORTS = {
    "football": Sport(weights="weights/football-player-detection.pt", imgsz=1280,
                      people=("player", "goalkeeper", "referee"), mask_green=True,
                      ball_weights="weights/football-ball-detection.pt",
                      # The Bundesliga-trained model misses players taller than its training
                      # data (near touchline on this elevated camera); COCO finds them.
                      person_weights="yolo11s.pt",
                      pitch_weights="weights/football-pitch-detection.pt"),
    "basketball": Sport(weights="weights/basketball-yolo11s.pt", imgsz=960,
                        people=("player", "player-shooting", "referee"), mask_green=False,
                        shots=True),
}


@dataclass
class Evidence:
    """Everything pass 1 collects."""
    people: list = field(default_factory=list)   # (frame, raw_id, class, box, score)
    colors: list = field(default_factory=list)   # (frame, raw_id, Lab feature)
    occluded: set = field(default_factory=set)   # (frame, raw_id) colour samples >20% covered
    reads: dict = field(default_factory=lambda: defaultdict(list))  # raw_id -> [(frame, text, conf)]
    ball: dict = field(default_factory=dict)     # frame -> box
    ball_conf: dict = field(default_factory=dict)
    in_basket: set = field(default_factory=set)  # frames with a ball-in-basket detection
    shooting: set = field(default_factory=set)   # frames with a player-shooting pose
    rims: dict = field(default_factory=dict)     # frame -> [box, ...]
    cuts: list = field(default_factory=list)     # frames where a new camera shot starts
    warps: dict = field(default_factory=dict)    # frame -> 2x3 camera motion (BoT-SORT GMC)
    pitch: dict = field(default_factory=dict)    # frame -> (32x2 keypoints, 32 conf), every 5th


def _frames(cap, batch: int, limit: int):
    buf = []
    for _ in range(limit):
        ok, frame = cap.read()
        if not ok:
            break
        buf.append(frame)
        if len(buf) == batch:
            yield buf
            buf = []
    if buf:
        yield buf


def _crop(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    return frame[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]


def collect(video: Path, sport: Sport, args) -> tuple[Evidence, dict]:
    """Pass 1: detection + tracking + per-track evidence."""
    import torch
    from scenedetect import AdaptiveDetector
    from scenedetect import detect as detect_scenes
    from ultralytics import YOLO
    from ultralytics.engine.results import Boxes
    from ultralytics.utils.metrics import bbox_ioa

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), args.max_frames or 10**9)
    gpu = torch.cuda.is_available()
    # Class-agnostic NMS: one person must not get both a "player" and a "player-shooting"
    # (or "goalkeeper") box, which would spawn two tracks on the same body.
    predict = dict(imgsz=args.imgsz or sport.imgsz, conf=0.1, agnostic_nms=True, verbose=False,
                   quantize=16 if gpu else None)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def load(weights):  # a TensorRT .engine built next to the .pt (Colab notebook) wins
        p = ROOT / weights
        return YOLO(str(p.with_suffix(".engine") if p.with_suffix(".engine").exists() else
                        p if p.exists() else weights))  # bare names (yolo11s.pt) auto-download

    model = load(sport.weights)
    ball_model = load(sport.ball_weights) if sport.ball_weights else None
    person_model = load(sport.person_weights) if sport.person_weights else None
    pitch_model = load(sport.pitch_weights) if sport.pitch_weights else None
    names = model.names
    people_ids = [i for i, n in names.items() if n in sport.people]
    ball_ids = [i for i, n in names.items() if n in ("ball", "ball-in-basket")]
    rim_ids = [i for i, n in names.items() if n == "rim"]
    shooting_ids = [i for i, n in names.items() if n == "player-shooting"]
    tracker = make_tracker(fps)
    reader = None if args.no_jersey else JerseyReader(ROOT / "weights/jersey-number-yolo11n.pt", gpu=gpu)
    last_read: dict[int, int] = {}
    ocr_tries = Counter()
    ev, f = Evidence(), 0
    ev.cuts = [s.get_frames() for s, _ in detect_scenes(str(video), AdaptiveDetector(), end_time=total)[1:]]
    gmc_apply = tracker.gmc.apply

    def record_warp(img, dets=None):  # keep the camera motion BoT-SORT estimates anyway
        ev.warps[f] = warp = gmc_apply(img, dets)
        return warp

    tracker.gmc.apply = record_warp
    t0 = time.time()

    for batch in _frames(cap, args.batch, total):
        results = model.predict(batch, **predict)
        balls = ball_model.predict(batch, **predict) if ball_model else [None] * len(batch)
        persons = (person_model.predict(batch, **{**predict, "conf": 0.5, "classes": [0]})
                   if person_model else [None] * len(batch))
        for frame, res, bres, pres in zip(batch, results, balls, persons):
            boxes = res.boxes.cpu().numpy()
            cls = boxes.cls.astype(int)
            # People cut off by the bottom edge are foreground crowd (stand-mounted cameras)
            # or partial bodies whose kit colour and identity are unreliable.
            # ponytail: assumes a camera in the stands; use a court/pitch mask from keypoint
            # homography for other rigs or when bench staff still leak in.
            dets = boxes.data[np.isin(cls, people_ids) & (boxes.xyxy[:, 3] < 0.99 * height)]
            if pres is not None:  # COCO persons the sport model missed, tagged "person"
                extra = pres.boxes.cpu().numpy().data
                extra = extra[extra[:, 3] < 0.99 * height]
                if len(dets) and len(extra):
                    extra = extra[bbox_ioa(extra[:, :4], dets[:, :4], iou=True).max(axis=1) < 0.3]
                extra[:, 5] = PERSON
                dets = np.concatenate([dets, extra])
            tracks = tracker.update(Boxes(dets, frame.shape[:2]), frame)
            hidden = bbox_ioa(tracks[:, :4], tracks[:, :4]) if len(tracks) else np.zeros((0, 0))
            np.fill_diagonal(hidden, 0)  # [j, i]: share of box i covered by box j
            for (x1, y1, x2, y2, tid, score, c, _), cover in zip(tracks, hidden.T):
                tid, name, box = int(tid), names.get(int(c), "person"), (float(x1), float(y1), float(x2), float(y2))
                ev.people.append((f, tid, name, box, float(score)))
                if f % 3 == 0:  # kit colour of every role (referee colour is needed too)
                    crop = _crop(frame, box)
                    if crop.size:
                        ev.colors.append((f, tid, jersey_color(crop, sport.mask_green)))
                        if cover.max(initial=0) > 0.2:
                            ev.occluded.add((f, tid))
                # OCR budget per track: every 10th frame, <= 30 tries, stop at 6 good reads
                # (enough for the 3-vote rule); keeps OCR cost bounded on long tracks.
                # ponytail: one CPU OCR call per crop, basketball runs ~13 fps; batch the crops
                # of a frame or use onnxruntime-gpu if real-time throughput matters.
                if (reader and y2 - y1 >= 80 and f - last_read.get(tid, -99) >= 10
                        and ocr_tries[tid] < 30 and len(ev.reads[tid]) < 6):
                    last_read[tid] = f
                    ocr_tries[tid] += 1
                    ev.reads[tid] += [(f, *r) for r in reader.read(_crop(frame, box))]
            if pitch_model is not None and f % 5 == 0:
                kp = pitch_model.predict(frame, imgsz=640, conf=0.3, verbose=False)[0].keypoints
                if kp is not None and kp.conf is not None and len(kp.xy):
                    ev.pitch[f] = (kp.xy[0].cpu().numpy(), kp.conf[0].cpu().numpy())

            # Ball candidates from both detectors: on the football clip each finds the ball in
            # ~20% of frames and their union in ~31%. Keep the most confident one.
            # ponytail: confidence-only pick, a ball-shaped logo can beat an occluded real ball;
            # move to motion-gated selection (or a WASB temporal model) if logos enter the track.
            cands = [(boxes.xyxy[i], boxes.conf[i], names[cls[i]] == "ball-in-basket")
                     for i in np.flatnonzero(np.isin(cls, ball_ids))]
            if bres is not None:
                bb = bres.boxes.cpu().numpy()
                cands += [(bb.xyxy[i], bb.conf[i], False) for i in range(len(bb))]
            box, conf, in_basket = max(cands, key=lambda c: c[1], default=(None, 0.0, False))
            if conf >= 0.2:  # low-score boxes still feed the tracker, but a lone 0.1 "ball" is noise
                ev.ball[f], ev.ball_conf[f] = tuple(box), float(conf)
                if in_basket:
                    ev.in_basket.add(f)
            if rim_ids:
                rims = boxes.xyxy[np.isin(cls, rim_ids) & (boxes.conf > 0.3)]
                if len(rims):
                    ev.rims[f] = [tuple(r) for r in rims]
            if (np.isin(cls, shooting_ids) & (boxes.conf > 0.4)).any():
                ev.shooting.add(f)
            f += 1
        if f % 500 < args.batch:
            print(f"  pass 1: {f}/{total} frames, {f / (time.time() - t0):.1f} fps", flush=True)

    meta = dict(fps=fps, frames=f, width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), detect_track_s=time.time() - t0)
    cap.release()
    return ev, meta


def _nearest_rim(rims, ball):
    return min(rims, key=lambda r: abs(r[0] + r[2] - ball[0] - ball[2])) if ball else rims[0]


def detect_shots(ev: Evidence, ball: dict, fps: float, frames: int):
    """Run the shot state machine over the cleaned ball track and the nearest rim.

    An undetected rim is carried through the camera motion for up to 3 s (the rim detector
    misses it in many frames on low cameras); a scene cut drops any half-seen attempt.
    """
    det, shots, rim, rim_seen, cuts = ShotDetector(fps), [], None, -10**9, set(ev.cuts)
    for f in range(frames):
        if f in cuts:
            shots, det, rim = shots + det.shots, ShotDetector(fps), None
        b = ball.get(f)
        if f in ev.rims:
            rim, rim_seen = _nearest_rim(ev.rims[f], b), f
        elif rim is not None and f - rim_seen <= 3 * fps:
            rim = warp_box(rim, ev.warps.get(f, np.eye(2, 3)))
        else:
            rim = None
        centre = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) if b else None
        det.update(f, centre, rim, ball_in_basket=f in ev.in_basket, shooting=f in ev.shooting)
    return shots + det.shots


def _homographies(ev: Evidence, frames: int) -> dict:
    """Image -> pitch homography per frame: fitted on keypoint frames (every 5th), smoothed
    with an EMA, held for at most 10 frames, reset at scene cuts."""
    out, H, fitted, cuts = {}, None, -10**9, set(ev.cuts)
    for f in range(frames):
        if f in cuts:
            H = None
        new = fit_homography(*ev.pitch[f]) if f in ev.pitch else None
        if new is not None:
            new = new / new[2, 2]
            H, fitted = (new if H is None else 0.5 * H + 0.5 * new), f
        if H is not None and f - fitted <= 10:
            out[f] = H
    return out


def _on_pitch(p, margin: float = 300) -> bool:
    return -margin <= p[0] <= L + margin and -margin <= p[1] <= W + margin


def _pitch_xy(H, x, y):
    return None if H is None else tuple(round(float(v)) for v in to_pitch([(x, y)], H)[0])


def _majority_role(people) -> dict:
    """Most frequent class per track ("player-shooting" is a player pose, not a role)."""
    votes = defaultdict(Counter)
    for _, tid, name, _, _ in people:
        votes[tid]["player" if name == "player-shooting" else name] += 1
    return {t: v.most_common(1)[0][0] for t, v in votes.items()}


def decide(ev: Evidence, sport: Sport, meta: dict) -> tuple[dict, dict]:
    """Whole-clip decisions -> (records per frame, summary)."""
    fps, frames = meta["fps"], meta["frames"]

    # Teams: KMeans on outfield-player colours. A track whose kit colour then changes for
    # good was swapped onto another player in an occlusion: split it there, so the new
    # person gets a new ID, and vote team / role / number per segment. Track keys become
    # (tracker id, segment index).
    obs_role = _majority_role(ev.people)
    fit = [(t, c) for _, t, c in ev.colors if obs_role.get(t) == "player"]  # confirmed players
    obs = [(f, t, c) for f, t, c in ev.colors if obs_role.get(t) in ("player", "person")]
    centres, labels, cuts = np.zeros((2, 3)), [], {}
    if len({t for t, _ in fit}) >= 2:
        _, centres = assign_teams(np.array([t for t, _ in fit]), np.array([c for _, c in fit]))
        feats = np.array([c for *_, c in obs])
        labels = label_teams(feats, centres)
        per_track = defaultdict(list)
        for (f, t, _), lab in zip(obs, labels):
            per_track[t].append((f, int(lab), (f, t) in ev.occluded))
        cuts = {t: team_switches(*zip(*seq)) for t, seq in per_track.items()}

    breaks = {t: sorted(set(c) | set(ev.cuts)) for t, c in cuts.items()}  # + scene cuts

    def seg(t, f):
        return t, bisect_right(breaks.get(t, ev.cuts), f)

    people = [(f, seg(t, f), name, box, s) for f, t, name, box, s in ev.people]
    reads = defaultdict(list)
    for t, rs in ev.reads.items():
        for f, text, conf in rs:
            reads[seg(t, f)].append((text, conf))
    team_votes = defaultdict(Counter)
    for (f, t, _), lab in zip(obs, labels):
        team_votes[seg(t, f)][int(lab)] += 1
    team = {s: v.most_common(1)[0][0] for s, v in team_votes.items()}
    role = _majority_role(people)

    # COCO finds every person, the sport model knows who is a player: a mostly-COCO track
    # counts only if the sport model also detected it on >= 10% of its frames. Coaches and
    # substitutes in the dugout never get a sport-model detection.
    support = defaultdict(Counter)
    for _, s, name, _, _ in people:
        support[s][name == "person"] += 1
    for s, n in support.items():
        if role[s] == "person":
            role[s] = "player" if n[False] >= 0.1 * (n[False] + n[True]) else "other"

    # "Players" whose kit matches neither team are coaches, bench staff, spectators, or
    # referees the detector missed: referee-coloured ones become referees, the rest go.
    seg_feats = defaultdict(list)
    for f, t, c in obs:
        seg_feats[seg(t, f)].append(c)
    ref_kit = [c for _, t, c in ev.colors if obs_role.get(t) == "referee"]
    outliers = kit_outliers(seg_feats, centres) if team else set()
    for s in outliers:
        kit = np.median(seg_feats[s], axis=0)
        is_ref = bool(ref_kit) and (kit_distance(kit, np.median(ref_kit, axis=0))
                                    < min(kit_distance(kit, c) for c in centres))
        role[s] = "referee" if is_ref else "other"
        team.pop(s, None)
    seen = Counter(s for _, s, *_ in people)
    keep = {s for s, n in seen.items() if n >= fps and role[s] != "other"}  # < 1 s: flicker

    # With a pitch homography, people whose feet stay off the pitch (staff, subs, crowd) go.
    homographies = _homographies(ev, frames)
    feet = defaultdict(list)
    for f, s, _, box, _ in people:
        if s in keep and f % 5 == 0 and f in homographies:
            feet[s].append(to_pitch([((box[0] + box[2]) / 2, box[3])], homographies[f])[0])
    off = {s for s, p in feet.items() if len(p) >= 3 and not _on_pitch(np.median(p, axis=0))}
    keep -= off

    # Goalkeepers wear their own kit: give them the team whose outfield players stand closest
    # (mean x in the same frame), majority-voted over the track.
    # ponytail: image-x proximity, wrong when only attackers are near the keeper; use a pitch
    # homography (which goal the keeper defends) if goalkeeper teams are misassigned.
    team_x, gk_x = defaultdict(lambda: ([], [])), defaultdict(list)
    for f, tid, _, box, _ in people:
        x = (box[0] + box[2]) / 2
        if tid in team and role.get(tid) == "player":
            team_x[f][team[tid]].append(x)
        elif role.get(tid) == "goalkeeper":
            gk_x[tid].append((f, x))
    for gk, positions in gk_x.items():
        side = Counter(int(abs(np.mean(team_x[f][1]) - x) < abs(np.mean(team_x[f][0]) - x))
                       for f, x in positions if all(team_x[f]))
        if side:
            team[gk] = side.most_common(1)[0][0]

    # Persistent IDs: tracklets with the same team + jersey number that are never on screen
    # together are one player (re-identification after leaving the frame). Then compact IDs
    # in order of first appearance; the ball is track 0.
    first, last = {}, {}
    for f, tid, *_ in people:
        if tid in keep:
            first.setdefault(tid, f)
            last[tid] = f
    numbers = {t: vote_number(reads[t]) for t in keep if reads.get(t)}
    alias = stitch_by_number({t: (first[t], last[t], team.get(t), numbers.get(t))
                              for t in keep if role[t] != "referee"})
    canon = {t: alias.get(t, t) for t in keep}
    rank = {c: i + 1 for i, c in enumerate(sorted(set(canon.values()), key=lambda c: (first[c], c)))}
    new_id = {t: rank[canon[t]] for t in keep}
    members = defaultdict(list)
    for t in keep:
        members[canon[t]].append(t)

    ball = fill_ball_gaps(ev.ball, max_gap=int(0.4 * fps), max_speed=0.04 * meta["width"],
                          breaks=ev.cuts)
    shots = detect_shots(ev, ball, fps, frames) if sport.shots else []

    records = defaultdict(list)

    def add(f, track_id, class_name, bbox, confidence=None, team_id=None, jersey_number=None, at=None):
        records[f].append(dict(track_id=track_id, class_name=class_name, bbox=bbox, confidence=confidence,
                               team_id=team_id, jersey_number=jersey_number,
                               pitch_xy=at and _pitch_xy(homographies.get(f), *at)))

    for f, tid, _, box, score in people:
        if tid in keep:
            r = role[tid]
            add(f, new_id[tid], r, box, score,
                None if r == "referee" or tid not in team else TEAM_NAMES[team[tid]],
                numbers.get(tid), ((box[0] + box[2]) / 2, box[3]))
    for f, box in ball.items():
        add(f, 0, "ball", box, ev.ball_conf.get(f), at=((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
    for f, rims in ev.rims.items():
        for box in rims:
            add(f, -1, "rim", box)

    summary = dict(
        meta=meta,
        identity=dict(raw_tracks=len({t for _, t, *_ in ev.people}), kept_tracks=len(keep),
                      team_switch_splits=sum(len(c) for c in cuts.values()),
                      kit_outliers=len(outliers),
                      number_merges=len(alias), identities=len(rank)),
        teams={TEAM_NAMES[k]: dict(jersey_color_bgr=lab_to_bgr(centres[k]),
                                   tracks=sum(1 for t in keep if team.get(t) == k)) for k in (0, 1)},
        tracks=[dict(track_id=rank[c], class_name=role[c],
                     team_id=TEAM_NAMES[team[c]] if c in team and role[c] != "referee" else None,
                     jersey_number=numbers.get(c), first_frame=first[c],
                     frames=sum(seen[t] for t in ms), tracklets=len(ms))
                for c, ms in sorted(members.items(), key=lambda kv: rank[kv[0]])],
        shots=[dict(shot_id=i + 1, start_frame=s.start_frame, end_frame=s.end_frame,
                    start_time=round(s.start_frame / fps, 2), end_time=round(s.end_frame / fps, 2),
                    outcome=s.outcome) for i, s in enumerate(shots)],
        shot_totals=dict(attempts=len(shots), made=sum(s.outcome == "made" for s in shots),
                         missed=sum(s.outcome == "missed" for s in shots)),
    )
    return records, summary


def _shot_hud(shots: list[dict], f: int, fps: float):
    done = [s for s in shots if s["end_frame"] <= f]
    made = sum(s["outcome"] == "made" for s in done)
    line = (f"SHOTS {len(done)}   MADE {made}   MISSED {len(done) - made}", None)
    banner = None
    for s in shots:
        if s["start_frame"] <= f < s["end_frame"]:
            banner = ("SHOT DETECTED", (0, 200, 255))
        elif s["end_frame"] <= f < s["end_frame"] + 2 * fps:
            banner = (f"RESULT: {s['outcome'].upper()}",
                      (60, 200, 60) if s["outcome"] == "made" else (60, 60, 230))
    return line, banner


CSV_FIELDS = ["frame_id", "timestamp", "track_id", "class_name", "x_min", "y_min", "x_max",
              "y_max", "confidence", "team_id", "jersey_number", "event", "shot_outcome",
              "pitch_x", "pitch_y"]


def render(video: Path, records: dict, summary: dict, sport: Sport, out_dir: Path, stem: str) -> None:
    """Pass 2: draw + encode the annotated video and write the per-frame CSV."""
    meta, fps = summary["meta"], summary["meta"]["fps"]
    colors = {k: vivid(v["jersey_color_bgr"]) for k, v in summary["teams"].items()}
    events = defaultdict(list)
    for s in summary["shots"]:
        events[s["start_frame"]].append(("shot_attempt", None))
        events[s["end_frame"]].append(("shot_result", s["outcome"]))

    cap = cv2.VideoCapture(str(video))
    writer = VideoWriter(out_dir / f"{stem}_annotated.mp4", meta["width"], meta["height"], fps)
    t0 = time.time()
    with open(out_dir / f"{stem}_tracks.csv", "w", newline="") as fh:
        rows = csv.writer(fh)
        rows.writerow(CSV_FIELDS)
        for f in range(meta["frames"]):
            ok, frame = cap.read()
            if not ok:
                break
            ts = round(f / fps, 3)
            objs = records.get(f, [])
            for o in objs:
                rows.writerow([f, ts, o["track_id"], o["class_name"], *(round(float(v), 1) for v in o["bbox"]),
                               "" if o["confidence"] is None else round(o["confidence"], 3),
                               o["team_id"] or "", o["jersey_number"] or "", "", "",
                               *(o["pitch_xy"] or ("", ""))])
            for kind, outcome in events.get(f, []):
                rows.writerow([f, ts, -1, "event", "", "", "", "", "", "", "", kind, outcome or "", "", ""])

            draw_objects(frame, objs, colors)
            if any(o["pitch_xy"] for o in objs):
                draw_radar(frame, objs, colors)
            count = Counter(o["team_id"] for o in objs if o["class_name"] in ("player", "goalkeeper"))
            lines = [(f"TEAM A  {count['A']} on screen", colors["A"]),
                     (f"TEAM B  {count['B']} on screen", colors["B"])]
            banner = None
            if sport.shots:
                line, banner = _shot_hud(summary["shots"], f, fps)
                lines.append(line)
            draw_hud(frame, lines + [(f"frame {f}  t={ts:.1f}s", None)], banner)
            writer.write(frame)
    writer.close()
    cap.release()
    summary["meta"]["render_s"] = time.time() - t0


def run(sport_name: str, video: Path, out_dir: Path, args) -> dict:
    sport = SPORTS[sport_name]
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video.stem
    cache = out_dir / f"{stem}_evidence.pkl"  # pass-1 output: re-decide without re-detecting
    if args.reuse and cache.exists():
        print(f"[{sport_name}] reusing pass-1 evidence {cache}", flush=True)
        ev, meta = pickle.loads(cache.read_bytes())
    else:
        print(f"[{sport_name}] pass 1: detection + tracking on {video}", flush=True)
        ev, meta = collect(video, sport, args)
        cache.write_bytes(pickle.dumps((ev, meta)))
    print(f"[{sport_name}] whole-clip decisions (teams, numbers, ball, shots)", flush=True)
    records, summary = decide(ev, sport, meta)
    print(f"[{sport_name}] pass 2: rendering", flush=True)
    render(video, records, summary, sport, out_dir, stem)
    summary = dict(sport=sport_name, source=str(video), **summary)
    (out_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"[{sport_name}] done -> {out_dir}", flush=True)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m sportsvision", description=__doc__.split("\n")[0])
    p.add_argument("sport", choices=sorted(SPORTS))
    p.add_argument("source", type=Path, help="input video")
    p.add_argument("--out", type=Path, default=Path("outputs"))
    p.add_argument("--imgsz", type=int, help="override the sport's inference size")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max-frames", type=int, default=0, help="stop early (smoke tests)")
    p.add_argument("--no-jersey", action="store_true", help="skip jersey-number recognition")
    p.add_argument("--reuse", action="store_true",
                   help="reuse <clip>_evidence.pkl from an earlier run: skip detection, re-decide + render")
    args = p.parse_args(argv)
    run(args.sport, args.source, args.out, args)


if __name__ == "__main__":
    main()
