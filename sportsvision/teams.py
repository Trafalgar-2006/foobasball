"""Unsupervised team classification from jersey colour.

Each player crop is reduced to the median Lab colour of its torso (grass pixels masked out),
the observations are clustered with KMeans(k=2), and every track takes the majority label of
its observations, so a few occluded or blurred crops cannot flip a player's team.
"""
import cv2
import numpy as np
from sklearn.cluster import KMeans

# OpenCV HSV hue is 0-180; pitch green sits around 60.
GREEN_HUE, MIN_GREEN_SAT = (35, 85), 50
# Sun vs shadow moves lightness (L) far more than kit chroma (a, b): on the U18 clip a
# shadowed red player was as dark as the blue team. Down-weight L so clusters follow kit
# hue; it still separates white-vs-black kits, whose L gap is ~150.
LAB_WEIGHTS = np.array([0.25, 1.0, 1.0])


def jersey_color(crop_bgr: np.ndarray, mask_green: bool = True) -> np.ndarray:
    """Median Lab colour of the torso region of a player box."""
    # ponytail: one kit colour per crop, fails when both kits share a hue or under strongly
    # coloured light; switch to SigLIP+UMAP embeddings (roboflow/sports TeamClassifier) then.
    h, w = crop_bgr.shape[:2]
    torso = crop_bgr[int(h * 0.15):int(h * 0.55), int(w * 0.2):int(w * 0.8)]
    lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    if mask_green:
        hue, sat = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV).reshape(-1, 3)[:, :2].T
        keep = ~((hue >= GREEN_HUE[0]) & (hue <= GREEN_HUE[1]) & (sat > MIN_GREEN_SAT))
        if keep.sum() >= 0.1 * keep.size:  # all-green crop: fall back to every pixel
            lab = lab[keep]
    return np.median(lab, axis=0)


def assign_teams(track_ids: np.ndarray, feats: np.ndarray, seed: int = 0):
    """Cluster observations into 2 teams; return ({track_id: team}, team centroids in Lab).

    Teams are ordered by centroid (L, a, b) so team 0/1 do not swap between runs.
    """
    km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(feats * LAB_WEIGHTS)
    centres = km.cluster_centers_ / LAB_WEIGHTS
    order = np.lexsort(centres.T[::-1])  # primary key: L
    labels = np.argsort(order)[km.labels_]
    teams = {int(t): int(np.bincount(labels[track_ids == t], minlength=2).argmax())
             for t in np.unique(track_ids)}
    return teams, centres[order]


def kit_distance(a, b) -> float:
    """Distance between two Lab kit colours in the clustering metric (L down-weighted)."""
    return float(np.linalg.norm((np.asarray(a, float) - np.asarray(b, float)) * LAB_WEIGHTS))


def kit_outliers(feats_by_track: dict, centres: np.ndarray, factor: float = 3.0,
                 floor: float = 12.0) -> set:
    """Tracks whose median kit colour is far from both team centroids: coaches, bench staff,
    spectators, or referees the detector called "player".

    Threshold: max(floor, factor x the median track-to-centroid distance) in the clustering
    metric, so it adapts to how tight the two kit clusters are on the clip.
    """
    if not feats_by_track:
        return set()
    ids = list(feats_by_track)
    medians = np.array([np.median(feats_by_track[t], axis=0) for t in ids])
    dist = np.linalg.norm((medians[:, None] - centres[None]) * LAB_WEIGHTS, axis=2).min(axis=1)
    threshold = max(floor, factor * float(np.median(dist)))
    return {t for t, d in zip(ids, dist) if d > threshold}


def label_teams(feats: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Team (0/1) of each colour observation: nearest centroid in the clustering metric."""
    d = np.linalg.norm((feats[:, None, :] - centres[None]) * LAB_WEIGHTS, axis=2)
    return d.argmin(axis=1)


def team_switches(frames: list[int], labels: list[int], occluded=None, min_run: int = 8,
                  near: int = 3) -> list[int]:
    """Frames where one track's kit colour changes for good -> the tracker swapped people.

    ``frames`` / ``labels``: per-observation team labels of a single track, in frame order.
    Runs shorter than ``min_run`` observations (~1 s at one sample per 3 frames) are noise
    and are absorbed into a neighbouring run. A remaining change of label is a split point
    only if the box overlapped another box within ``near`` observations of it (trackers swap
    people during occlusions; sun-to-shadow changes happen without one).
    """
    runs = []  # [label, first observation index, length]
    for i, lab in enumerate(labels):
        if runs and runs[-1][0] == lab:
            runs[-1][2] += 1
        else:
            runs.append([lab, i, 1])
    for k, run in enumerate(runs):
        if run[2] < min_run and len(runs) > 1:
            run[0] = runs[k - 1][0] if k else runs[1][0]
    splits, current = [], runs[0][0] if runs else None
    for lab, i, _ in runs[1:]:
        if lab != current and (occluded is None or any(occluded[max(0, i - near):i + near])):
            splits.append(frames[i])
            current = lab
    return splits
