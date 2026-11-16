from sportsvision.jersey import stitch_by_number


def test_same_team_and_number_tracklets_become_one_identity():
    spans = {1: (0, 100, "A", "23"), 7: (300, 400, "A", "23"), 9: (500, 600, "A", "23")}
    assert stitch_by_number(spans) == {7: 1, 9: 1}


def test_overlapping_tracklets_are_different_players():
    # a misread: two players on screen at once cannot be the same person
    spans = {1: (0, 100, "A", "23"), 2: (50, 150, "A", "23")}
    assert stitch_by_number(spans) == {}


def test_number_is_only_shared_within_a_team():
    spans = {1: (0, 100, "A", "5"), 2: (200, 300, "B", "5")}
    assert stitch_by_number(spans) == {}


def test_tracks_without_number_or_team_are_left_alone():
    spans = {1: (0, 100, "A", None), 2: (200, 300, "A", None), 3: (400, 500, None, "8"),
             4: (600, 700, None, "8")}
    assert stitch_by_number(spans) == {}
