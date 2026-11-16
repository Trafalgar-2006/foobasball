import numpy as np
import pytest

from sportsvision.geometry import PITCH, fit_homography, to_pitch, warp_box
from sportsvision.tracking import fill_ball_gaps


def box(x, y, r=5):
    return (x - r, y - r, x + r, y + r)


def test_ball_gap_across_a_scene_cut_is_not_filled():
    out = fill_ball_gaps({0: box(0, 0), 4: box(40, 80)}, max_gap=5, max_speed=50, breaks=[2])
    assert sorted(out) == [0, 4]


def test_warp_box_applies_camera_motion():
    shift = np.array([[1.0, 0, 30], [0, 1.0, -10]])  # camera panned: content moved +30, -10
    assert warp_box((100, 100, 140, 120), shift) == pytest.approx((130, 90, 170, 110))


def project(H, pts):
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:]


def test_homography_recovered_from_pitch_keypoints():
    true = np.array([[0.08, 0.01, 200.0], [0.002, 0.05, 150.0], [0.0, 0.00002, 1.0]])  # pitch -> image
    image_pts = project(true, PITCH)
    conf = np.ones(len(PITCH))
    H = fit_homography(image_pts, conf)  # image -> pitch
    assert to_pitch(image_pts[:5], H) == pytest.approx(PITCH[:5], abs=1.0)


def test_too_few_or_collinear_keypoints_give_no_homography():
    image_pts = np.random.default_rng(0).uniform(0, 1000, (32, 2))
    conf = np.zeros(32)
    conf[:3] = 1  # only 3 confident points
    assert fit_homography(image_pts, conf) is None
    conf[13:17] = 1  # halfway-line points (14-17) are collinear on the pitch
    conf[:3] = 0
    assert fit_homography(image_pts, conf) is None
