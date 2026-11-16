"""Pitch registration (image -> pitch homography) and camera-motion helpers.

The 32-keypoint template follows roboflow/sports `SoccerPitchConfiguration` (MIT); its
order matches Roboflow's `football-pitch-detection` keypoint model. Units: cm, 120 x 70 m.
"""
import cv2
import numpy as np

L, W = 12000, 7000                   # pitch length, width
PB_L, PB_W, GB_L, GB_W = 2015, 4100, 550, 1832  # penalty box, goal box
CC_R, SPOT = 915, 1100               # centre-circle radius, penalty spot
_y = [0, (W - PB_W) / 2, (W - GB_W) / 2, (W + GB_W) / 2, (W + PB_W) / 2, W]
PITCH = np.array(
    [(0, y) for y in _y]                                             # 1-6 goal line
    + [(GB_L, _y[2]), (GB_L, _y[3]), (SPOT, W / 2)]                 # 7-9
    + [(PB_L, y) for y in _y[1:5]]                                  # 10-13
    + [(L / 2, 0), (L / 2, W / 2 - CC_R), (L / 2, W / 2 + CC_R), (L / 2, W)]  # 14-17
    + [(L - PB_L, y) for y in _y[1:5]]                              # 18-21
    + [(L - SPOT, W / 2), (L - GB_L, _y[2]), (L - GB_L, _y[3])]     # 22-24
    + [(L, y) for y in _y]                                           # 25-30
    + [(L / 2 - CC_R, W / 2), (L / 2 + CC_R, W / 2)],               # 31-32
    np.float32)
EDGES = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8), (10, 11), (11, 12), (12, 13),
         (14, 15), (15, 16), (16, 17), (18, 19), (19, 20), (20, 21), (23, 24), (25, 26),
         (26, 27), (27, 28), (28, 29), (29, 30), (1, 14), (2, 10), (3, 7), (4, 8), (5, 13),
         (6, 17), (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30)]


def fit_homography(image_pts: np.ndarray, conf: np.ndarray, min_conf: float = 0.5):
    """Image -> pitch homography from confident keypoints (RANSAC, 50 cm); None when the
    points can't pin it down: fewer than 4, or all on one pitch line (e.g. halfway line)."""
    ok = (conf > min_conf) & (image_pts[:, 0] > 1) & (image_pts[:, 1] > 1)
    if ok.sum() < 4:
        return None
    dst = PITCH[ok]
    spread = np.linalg.svd(dst - dst.mean(axis=0), compute_uv=False)
    if spread[1] < 0.05 * spread[0]:
        return None
    H, inliers = cv2.findHomography(image_pts[ok].astype(np.float32), dst, cv2.RANSAC, 50.0)
    return H if H is not None and inliers.sum() >= 4 else None


def to_pitch(points, H: np.ndarray) -> np.ndarray:
    """Image points (N x 2) -> pitch coordinates in cm."""
    pts = np.asarray(points, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def warp_box(box, warp: np.ndarray) -> tuple:
    """Move a box by a 2x3 camera-motion warp (previous frame -> current frame)."""
    corners = np.array([[box[0], box[1], 1.0], [box[2], box[3], 1.0]]) @ np.asarray(warp).T
    return tuple(float(v) for v in corners.ravel())
