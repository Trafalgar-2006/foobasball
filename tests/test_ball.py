import pytest

from sportsvision.tracking import fill_ball_gaps


def box(x, y, r=5):
    return (x - r, y - r, x + r, y + r)


def test_short_gap_is_linearly_interpolated():
    out = fill_ball_gaps({0: box(0, 0), 4: box(40, 80)}, max_gap=5, max_speed=50)
    assert out[2] == pytest.approx(box(20, 40))
    assert sorted(out) == [0, 1, 2, 3, 4]


def test_long_gap_is_left_empty():
    out = fill_ball_gaps({0: box(0, 0), 20: box(10, 10)}, max_gap=5, max_speed=50)
    assert sorted(out) == [0, 20]


def test_teleporting_detection_is_dropped_as_outlier():
    track = {0: box(0, 0), 1: box(5, 0), 2: box(900, 500), 3: box(15, 0)}
    out = fill_ball_gaps(track, max_gap=5, max_speed=50)
    assert out[2] == pytest.approx(box(10, 0))
