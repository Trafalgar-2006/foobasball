from sportsvision.shots import ShotDetector

RIM = (100.0, 200.0, 140.0, 210.0)  # x1, y1, x2, y2 -> centre (120, 205), width 40


def run(path, rim=RIM, fps=30, in_basket=(), shooting=()):
    """Feed a ball path (list of (x, y) or None) and return emitted events."""
    det = ShotDetector(fps=fps)
    events = []
    for i, ball in enumerate(path):
        events += det.update(i, ball, rim, ball_in_basket=i in in_basket, shooting=i in shooting)
    return det, events


def drop(x0, x1, y0=120, y1=300, n=12):
    """Straight-line ball path from (x0, y0) to (x1, y1)."""
    return [(x0 + (x1 - x0) * t / (n - 1), y0 + (y1 - y0) * t / (n - 1)) for t in range(n)]


def test_ball_dropping_through_rim_is_a_made_shot():
    det, events = run(drop(120, 120))
    assert [e.kind for e in events] == ["attempt", "made"]
    assert [s.outcome for s in det.shots] == ["made"]


def test_ball_crossing_rim_height_outside_rim_is_a_miss():
    det, events = run(drop(150, 175))
    assert [e.kind for e in events] == ["attempt", "missed"]


def test_ball_below_rim_never_triggers_a_shot():
    det, events = run([(120, 260 + (i % 5)) for i in range(60)])
    assert events == []


def test_ball_above_rim_far_from_hoop_is_not_an_attempt():
    det, events = run(drop(600, 600))
    assert events == []


def test_rebound_right_after_a_shot_is_not_double_counted():
    det, events = run(drop(120, 120) + drop(120, 120))
    assert [e.kind for e in events] == ["attempt", "made"]


def test_second_shot_after_cooldown_is_counted():
    det, events = run(drop(120, 120) + [None] * 60 + drop(150, 175))
    assert [s.outcome for s in det.shots] == ["made", "missed"]


def test_ball_in_basket_detection_confirms_make():
    path = drop(120, 120)[:5] + [None] * 3 + [(120, 225)] * 2 + [None] * 5  # into the net
    det, events = run(path, in_basket={8, 9})
    assert [e.kind for e in events] == ["attempt", "made"]


def test_single_ball_in_basket_blip_is_not_a_make():
    path = [None] * 3 + [(120, 225)] + [None] * 120  # one detection, no arc ever seen
    det, events = run(path, in_basket={3})
    assert [e.kind for e in events] == ["attempt", "missed"]


def test_ball_in_basket_alone_triggers_attempt_when_ball_was_not_tracked():
    in_net = [(120, 225)] * 10  # just under the rim
    det, events = run(in_net, in_basket={4, 5})
    assert [e.kind for e in events] == ["attempt", "made"]


def test_ball_in_basket_far_from_the_rim_is_ignored():
    # e.g. a ball logo on the scorer's table classified as ball-in-basket every frame
    det, events = run([(900, 700)] * 120, in_basket=set(range(120)))
    assert events == []


def test_attempt_with_ball_lost_resolves_as_miss_after_timeout():
    det, events = run(drop(120, 120)[:5] + [None] * 120)
    assert [e.kind for e in events] == ["attempt", "missed"]


def test_shot_start_is_backdated_to_the_shooting_motion():
    path = [(300, 400)] * 10 + drop(120, 120)  # player rises to shoot at frames 0-4
    det, events = run(path, shooting=set(range(5)))
    assert det.shots[0].start_frame == 0
    assert det.shots[0].outcome == "made"


def test_shooting_pose_without_the_ball_reaching_the_rim_is_not_an_attempt():
    det, events = run([(300, 400)] * 120, shooting=set(range(5)))  # pump fake
    assert events == []
