"""Integration: whole-clip decisions from synthetic pass-1 evidence (no models, no video)."""
from sportsvision.pipeline import SPORTS, Evidence, decide

RED, BLUE = (90.0, 140.0, 145.0), (60.0, 130.0, 122.0)  # Lab kits measured on the U18 clip
FPS, FRAMES = 10, 60
META = dict(fps=FPS, frames=FRAMES, width=1920, height=1080, detect_track_s=1.0)


def evidence(players, referee=None):
    """players: {track_id: (kit, x)}; every track visible for the whole clip, side by side."""
    ev = Evidence()
    for f in range(FRAMES):
        for tid, (kit, x) in players.items():
            ev.people.append((f, tid, "player", (x, 400.0, x + 40.0, 520.0), 0.9))
            if f % 3 == 0:
                ev.colors.append((f, tid, kit))
        if referee:
            ev.people.append((f, referee, "referee", (1500.0, 400.0, 1540.0, 520.0), 0.9))
    return ev


def test_teams_referee_and_jersey_number_are_decided_per_track():
    ev = evidence({1: (RED, 100.0), 2: (RED, 300.0), 3: (BLUE, 500.0), 4: (BLUE, 700.0)}, referee=5)
    ev.reads[1] = [(0, "10", 0.9), (10, "10", 0.9), (20, "10", 0.8)]
    records, summary = decide(ev, SPORTS["football"], META)
    at = {r["bbox"][0]: r for r in records[30]}  # players by x position
    assert at[100]["team_id"] == at[300]["team_id"] != at[500]["team_id"] == at[700]["team_id"]
    assert at[1500]["class_name"] == "referee" and at[1500]["team_id"] is None
    assert at[100]["jersey_number"] == "10"
    assert len({r["track_id"] for r in records[30]}) == 5 == summary["identity"]["identities"]


def test_basketball_make_is_found_through_the_rim():
    ev = evidence({1: (RED, 100.0), 2: (BLUE, 500.0)})
    rim = (900.0, 300.0, 940.0, 310.0)
    for f in range(FRAMES):
        ev.rims[f] = [rim]
    for i, f in enumerate(range(20, 32)):  # ball drops straight through the rim
        y = 200.0 + i * 15
        ev.ball[f], ev.ball_conf[f] = (915.0, y, 925.0, y + 10), 0.9
    _, summary = decide(ev, SPORTS["basketball"], META)
    assert [s["outcome"] for s in summary["shots"]] == ["made"]
