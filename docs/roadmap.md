# Improvement roadmap

Ranked plan for making `sportsvision` better, built from what was **measured** on the two
assignment clips and from the improvement research in
[`research.md`](research.md) and [`research-improvements.md`](research-improvements.md).
Every item has a metric that proves it worked; nothing is done "because it should help".

Rules for every item:

* **Measure first.** Phase 0 builds the yardsticks; no tuning before them.
* **Smallest change that moves the metric** (ponytail). Items that don't move a metric get
  cut, not merged.
* **TDD at agreed seams.** Seams below are *proposals*; confirm each before writing its test.
  Pure decision logic gets red-green unit tests; models get metric gates; refactors get the
  golden-output check (same CSV before/after on fixed clips).

## 2-day execution plan (reuse first)

The deadline allows two working days, so every item below reuses an existing open-source
component rather than building one, and everything that needs a new detection pass goes into
**one** cached pass-1 run per clip (`--reuse` then re-decides and re-renders in minutes).

| When | Item | Reused component | Status |
|---|---|---|---|
| Day 1 | Evidence cache, `--reuse` | stdlib `pickle` | done |
| Day 1 | Q3 scene cuts: split tracks, no ball interpolation or shot carry-over across cuts | PySceneDetect `AdaptiveDetector` (BSD-3) | done |
| Day 1 | Q2 rim persistence through camera motion (≤3 s) | BoT-SORT's own GMC warps, recorded per frame | done |
| Day 1 | R1/R2 pitch homography, radar minimap, `pitch_x/y` in the CSV, on-pitch filter | Roboflow `football-pitch-detection` + roboflow/sports pitch template (MIT) + OpenCV RANSAC | done |
| Day 1 | appearance vectors for re-identification | Ultralytics `yolo11n-cls` `embed()` | tried and removed: teammates measured as close as the same player |
| Night 1 | full pass-1 runs with all new evidence | - | done |
| Day 2 | I1 offline tracklet merge with team / number / overlap cannot-links | port of gta-link's merge step (MIT) | dropped with the appearance signal; number-only stitching kept |
| Day 2 | Q1 kit-split tuning (461 splits) on cached evidence | - | done: a kit flip splits a track only during a box overlap |
| Day 2 | S1-S3 shots: score-bug frame miner, pre-labels, YOLO11s fine-tune, re-run, `eval_shots.py` | Grounding DINO / OWLv2 via Hugging Face `transformers` (Apache-2.0) | not done: the pre-labelling model download failed; README Future work #1 |
| Day 2 | README + assets from final outputs; ponytail ultra rewrite with golden-output check; commit | - | done |
| Day 3 | Colab T4 run (TensorRT FP16 optional), push, Drive upload, submission | Ultralytics `export(format="engine")` | pushed; notebook provided, full T4 run not timed |

Deferred beyond the deadline: tracking ground truth (E1/E2), one-detector football (F1),
basketball court registration (R3), SigLIP teams (T1), PARSeq fine-tune (J1).

## Baseline being improved

| Area | Measured now | Target signal |
|---|---|---|
| Basketball shots | 20 attempts / 5 made; make precision **0.40**; field-goal recall **2/14** (scoreboard ground truth) | FG recall, make precision |
| Root cause | ball never detected near the rim in **12/14** FG windows; rim visible **4-57%** of frames | ball-near-rim coverage |
| Identities | football 694 per 10 min; basketball 1,149 per quarter, **441 kit-colour splits (38% of IDs)**: the split rule over-fires | HOTA / IDF1 / ID switches |
| Football speed | 3 detectors, **~2.2 fps** on 4 GB (≈8 fps ceiling on a T4 even with TensorRT) | fps |
| Jersey numbers | off-the-shelf PARSeq on located number boxes, ≥3-vote | per-track number accuracy |
| Teams | Lab kit colour, k=2; no measured accuracy | per-track team accuracy |
| Geometry | none: bottom-edge crowd hack, goalkeeper side by image-x | - |

## Phase 0: yardsticks (prerequisite, ~1 day)

| ID | Work | Seam (proposed) | Done when |
|---|---|---|---|
| E1 | Label ground truth in **CVAT track mode** (linear interpolation between keyframes): 2-3 dense 60 s windows per sport that contain exits/re-entries, plus team and jersey attributes per track | - | MOT-format GT committed under `eval/` |
| E2 | `tools/eval_tracking.py`: CSV → MOT rows → **TrackEval** (MIT) HOTA / IDF1 / AssA, plus team and number accuracy per GT track | `to_mot_rows(records)` pure; TrackEval itself not unit-tested | baseline numbers in README |
| E3 | **Evidence cache**: pass 1 pickles `Evidence`, `--evidence` reuses it, so every decision-layer change below re-runs in seconds instead of hours | `save/load round-trip equals` | re-run of `decide()` on a full clip < 1 min |
| E4 | CI on GitHub Actions (CPU): pytest + plumbing smoke on a synthetic `ffmpeg testsrc` clip with the committed basketball model, `--no-jersey --max-frames 30`, checks CSV header and JSON keys | smoke = integration test | green badge |
| E5 | `decide()` integration tests from synthetic `Evidence` (largest coverage gap: segments, outliers, keepers, stitching, records) | `decide(ev, sport, meta)` | 90%+ line coverage on decision modules |

## Phase 1: quick wins, no training (~2 days)

| ID | Work | Evidence | Seam (proposed) | Verify |
|---|---|---|---|---|
| Q1 | **Stop the kit-split over-firing**: raise `min_run` and only split where the colour flip starts during a box overlap with another track (that is when trackers swap) | 441 splits = 38% of IDs | `team_switches(frames, labels, occluded, min_run)` | splits ↓, IDF1 ↑ on E1 windows (cache E3 makes this a parameter sweep) |
| Q2 | **Rim persistence**: carry the last rim box through camera motion (BoT-SORT's GMC warp, captured per frame) for ≤3 s instead of dropping it after 0.5 s | rim visible only 4-57% of frames | `warp_box(box, H)` pure | rim coverage in the 14 FG windows ↑; FG recall (eval_shots) |
| Q3 | **Scene cuts**: PySceneDetect `AdaptiveDetector` (BSD-3, CPU; F1 91.6 on BBC) → reset tracker IDs and shot state at cuts; replays/close-ups gated by "too few people / no rim" | no cut handling today | `cut_frames(video)` adapter, integration test on a synthetic clip with one hard cut | no ID survives a cut in GT windows |
| Q4 | **Batched GPU OCR**: PARSeq via onnxruntime-gpu, all crops of a frame in one call, read every 5th frame (Roboflow recipe) | CPU PARSeq ~15 ms/crop | `read_batch(crops) == [read(c) for c]` | same numbers (golden), fps ↑ |
| Q5 | **TensorRT FP16** engines built on the T4 (`format=engine, quantize=16, imgsz=1280, batch=4`); FP16 measured 1.85-2.55x over ONNX FP16 with no accuracy loss. Skip INT8 (0.9-1.2x more, -3.4 to -7.6 mAP50-95) | - | config only | test-split mAP within 0.5 pt, fps ↑ |

## Phase 2: identity (~3 days)

| ID | Work | Evidence | Seam (proposed) | Verify |
|---|---|---|---|---|
| I1 | **Offline tracklet merging, GTA-style** (MIT reference): per-tracklet appearance embedding (OSNet via torchreid, MIT, with ImageNet/Market weights; *not* the SportsMOT-trained ones, CC BY-NC), merge by cosine distance < 0.4 with cannot-link rules: temporal overlap, different team, conflicting numbers. Spatial term **off** (panning camera). `stitch_by_number` becomes one cannot-link/must-link case | ByteTrack+GTA: +6.28 HOTA, +12.40 IDF1 on SportsMOT | `merge_tracklets(tracklets) -> alias` | identities ↓, IDF1 ↑ on E1 |
| I2 | **Swap veto** in BoT-SORT (subclass `get_dists`, ~15 lines): cost = 1 when the detection's running kit label contradicts the track's team. Today appearance is fused as `min(IoU, emb)` and can never block a swap | 441 splits | `veto(dists, track_teams, det_teams)` | ID switches ↓; Q1 splits ↓ further |

Order: I1 first (offline, bigger published gain, doesn't touch the tracker), I2 only if E1
still shows cross-team swaps.

## Phase 3: basketball shots: fix with data (~4 days, ~4 h of human review)

No open-vocabulary model has been evaluated on broadcast rims or balls; the research and
Roboflow's own 2025 pipeline both point to **in-domain labels**.

| ID | Work | Seam (proposed) | Verify |
|---|---|---|---|
| S1 | **Mine frames with the scoreboard**: every +2/+3 score change (already read by `eval_shots.py`) marks a make; take the 6 s before it near the hoop, plus random negatives: ~1,000 frames | `windows_from_baskets(baskets, fps)` | frame list covers all 14 FG windows |
| S2 | **Pre-label** with Grounding DINO-tiny / OWLv2 (Apache-2.0) prompts "basketball", "basketball hoop", SAM 2.1 (Apache-2.0) for tight boxes; human review in CVAT. Avoid SAM3 (custom licence) | - | reviewed YOLO labels |
| S3 | **Fine-tune YOLO11s @1280 on the T4** on NBA data + the new frames; keep the shot state machine unchanged | metric gate | in-domain ball/rim mAP; ball-near-rim coverage in FG windows; FG recall and make precision via `eval_shots.py` |
| S4 | *Optional*: WASB temporal ball model (MIT code; basketball F1 82.6 vs 57.2 single-frame) as a second ball source, inference-only (its training code and NBA data licence are unpublished) | adapter returning ball candidates | ball coverage ↑ |

Held back: score-bug or shot-clock fusion *deciding* outcomes. It would make the HUD right
but stop being a vision result, and it would make `eval_shots.py` circular; keep the
scoreboard as ground truth and as the label miner (S1).

## Phase 4: one football detector + speed (~3 days)

| ID | Work | Verify |
|---|---|---|
| F1 | **Replace three detectors with one**: pseudo-label match frames with today's ensemble (sport model + COCO persons + ball model, support-gated), review a subset, fine-tune **YOLO11m @1280** (classes player/goalkeeper/referee/ball). Data: Roboflow football dataset copies (CC BY 4.0) + own frames; not SoccerNet (NDA) | people recall on E1 ≥ ensemble; ball coverage ≥ union; fps |
| F2 | TensorRT FP16 for it (≈19 ms/frame on a T4 at 1280 vs ≈125 ms for today's stack) | fps on T4 |

## Phase 5: pitch / court registration (~4 days)

| ID | Work | Seam (proposed) | Verify |
|---|---|---|---|
| R1 | **Football pitch homography**: Roboflow `football-pitch-detection` (32 keypoints, AGPL via Ultralytics, already on gdown), `cv2.findHomography(RANSAC, 10 px)`, reprojection-error gate, ≥4 non-collinear points (halfway-line points alone are degenerate), EMA smoothing, fall back to GMC-propagating the last good H. Use the real pitch size, not the 120×70 m template (up to ~14% distance error) | `fit_homography(kps, template) -> H or None`; `smooth(Hs)` | synthetic projective transform recovered; collinear → None |
| R2 | Unlocks: **on-pitch filter** (replaces the bottom-edge crowd hack and most of the kit-outlier/support gating), **keeper side by pitch half**, **radar minimap**, speed/distance (only with smoothing: 0.3-0.6 m jitter = 7-15 m/s fake speed) | `on_field(points, H)` | dugout/bench FPs → 0 on E1 |
| R3 | **Basketball court**: fine-tune YOLO11s-pose on the court-keypoint dataset + 100-300 own frames, FIBA template (arc 6.75 m), `fliplr=0` | same seams | on-court filter; 2-vs-3-point label per attempt |

Note: a floor homography does not locate the rim (3.05 m up); rim stays a detector job (S3).

## Phase 6: only if the yardsticks ask for it

| ID | Work | Trigger |
|---|---|---|
| T1 | SigLIP + UMAP + KMeans (MIT code, Apache-2.0 weights) on ≤10 crops per tracklet | team accuracy on E1 < 95% or kits share a hue |
| J1 | Fine-tune PARSeq on *own* labelled number crops (+6.9 pts published for soccer); legibility classifier | number accuracy on E1 low. Do **not** reuse Koshkina's code/weights (CC BY-NC) or SoccerNet jersey data (no licence, NDA source) |

## Test plan

| Layer | What | Where it runs | Target |
|---|---|---|---|
| Unit (many, fast) | shot state machine, kit colour / teams / switches / outliers, jersey vote / stitching, ball gaps, score changes, every new pure seam above | CI, CPU | 90%+ lines on decision modules |
| Integration | `decide()` on synthetic `Evidence`; CSV/JSON schema from a synthetic clip; evidence cache round-trip | CI, CPU | every output field asserted |
| Model metrics | detector mAP on held-out splits; HOTA/IDF1/AssA, team and number accuracy on E1; shot FG recall / make precision on the scoreboard | GPU, offline | must not regress; gate merges |
| Golden end-to-end | 150-frame football + 300-frame basketball smoke, CSV compared line by line | GPU, before/after every refactor | identical |

Done: `decide()` integration tests, CI. Gap: no tracking ground truth (E1/E2).

## Licence register

| Asset | Licence | Use |
|---|---|---|
| Ultralytics, BoxMOT | AGPL-3.0 | ok (project is AGPL-3.0) |
| GTA reference, torchreid, TrackEval, WASB code, TrackNetV3, OnnxTR/PARSeq | MIT / Apache-2.0 | ok |
| Grounding DINO, OWLv2, SAM 2.1, SigLIP | Apache-2.0 | ok |
| PySceneDetect / TransNetV2 | BSD-3 / MIT | ok |
| PnLCalib | GPL-2.0+ code, CC BY 4.0 weights | ok (keyframes only, slow) |
| YOLO-World, T-DEED | GPL-3.0 | ok with AGPL |
| SportsMOT, PL-NBA, Koshkina jersey pipeline | CC BY-NC | **avoid** |
| SoccerNet (videos, jersey data) | NDA / unstated | **avoid** for shipped weights |
| SAM3 | custom "SAM License" | labelling only, not shipped |
| No-Bells-Just-Whistles | GPL-2.0-only | **avoid** (AGPL-incompatible) |
| sn-calibration, Deep-EIoU | no licence | **avoid** |

## Order of work

```
Phase 0 (E1-E5) ──> Phase 1 (Q1-Q5) ──┬──> Phase 2 identity (I1, then I2)
                                      ├──> Phase 3 shots data (S1-S3, S4 optional)
                                      └──> Phase 4 one detector (F1-F2) ──> Phase 5 registration (R1-R3)
Phase 6 only when Phase 0 metrics demand it.
```

## Deliberately not doing

INT8 (little speed, real accuracy loss); VLM jersey OCR (slow, weaker than a small
recogniser); sn-gamestate / TrackLab (GPL with old PyTorch pins, too heavy for a T4);
detector frame-skipping (loses the ball); AFLink (pedestrian-trained, no appearance);
score-bug fusion for outcomes (see Phase 3).
