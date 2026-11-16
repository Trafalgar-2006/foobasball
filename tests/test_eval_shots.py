import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "eval_shots", Path(__file__).resolve().parent.parent / "tools" / "eval_shots.py")
eval_shots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_shots)


def test_persistent_score_increase_is_one_basket():
    series = [(0, 0), (1, 0), (2, 2), (3, 2), (4, 2)]
    assert eval_shots.score_changes(series) == [(2, 2)]


def test_single_frame_ocr_glitch_is_ignored():
    series = [(0, 4), (1, 7), (2, 4), (3, 4), (4, None), (5, 4)]
    assert eval_shots.score_changes(series) == []


def test_free_throw_then_three_pointer():
    series = [(0, 10), (1, 11), (2, 11), (3, 11), (4, 14), (5, 14)]
    assert eval_shots.score_changes(series) == [(1, 1), (4, 3)]


def test_made_shots_match_score_changes_within_operator_lag():
    shots = [{"end_time": 10.0, "outcome": "made"}, {"end_time": 50.0, "outcome": "made"},
             {"end_time": 70.0, "outcome": "missed"}]
    baskets = [(12.5, 2), (90.0, 3)]  # second basket was never detected
    tp, matched = eval_shots.match(shots, baskets, lag=(-2, 8))
    assert tp == 1 and matched == [(10.0, 12.5)]
