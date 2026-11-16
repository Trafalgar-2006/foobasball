import numpy as np

from sportsvision.teams import assign_teams, jersey_color, kit_outliers, team_switches

ORANGE, BLUE, GRASS = (0, 110, 255), (200, 60, 20), (40, 140, 40)  # BGR


def crop(shirt_bgr, background_bgr=GRASS, h=60, w=30):
    """Player box: shirt in the torso area, grass everywhere else."""
    img = np.full((h, w, 3), background_bgr, np.uint8)
    img[int(h * 0.15):int(h * 0.55), int(w * 0.2):int(w * 0.8)] = shirt_bgr
    return img


def test_jersey_color_ignores_grass_around_the_player():
    orange, blue = jersey_color(crop(ORANGE)), jersey_color(crop(BLUE))
    grass = jersey_color(crop(GRASS, background_bgr=GRASS), mask_green=False)
    assert np.linalg.norm(orange - blue) > 50
    assert np.linalg.norm(orange - grass) > 50


def test_tracks_are_split_by_kit_with_majority_vote():
    rng = np.random.default_rng(0)
    ids, feats = [], []
    for track, kit in [(1, ORANGE), (2, ORANGE), (3, BLUE), (4, BLUE)]:
        for i in range(20):
            noisy = (i == 0)  # one occluded observation with the other team's colour
            colour = (BLUE if kit == ORANGE else ORANGE) if noisy else kit
            feats.append(jersey_color(crop(colour)) + rng.normal(0, 2, 3))
            ids.append(track)
    teams, _ = assign_teams(np.array(ids), np.array(feats))
    assert teams[1] == teams[2] != teams[3] == teams[4]


def test_shadowed_player_is_grouped_by_kit_hue_not_brightness():
    # Per-track median Lab measured on the U18 football clip: red kit in sun and shade,
    # blue kit. Track 2 is a red player in shadow, as dark as the blue team.
    red = {1: (85, 139, 146), 2: (61, 142, 140), 3: (101, 140, 143), 7: (90, 138, 145)}
    blue = {4: (53, 131, 121), 5: (50, 131, 121), 8: (63, 129, 124), 9: (65, 131, 122)}
    ids, feats = [], []
    for track, lab in {**red, **blue}.items():
        ids += [track] * 10
        feats += [lab] * 10
    teams, _ = assign_teams(np.array(ids), np.array(feats, float))
    assert len({teams[t] for t in red}) == 1
    assert len({teams[t] for t in blue}) == 1
    assert teams[2] != teams[4]


def test_team_ids_are_stable_regardless_of_input_order():
    feats = np.array([jersey_color(crop(c)) for c in (ORANGE, BLUE)] * 5)
    ids = np.array([1, 2] * 5)
    a, _ = assign_teams(ids, feats)
    b, _ = assign_teams(ids[::-1], feats[::-1])
    assert a == b


def frames_for(labels):
    return [3 * i for i in range(len(labels))]  # colour sampled every 3rd frame


def test_steady_kit_never_splits():
    labels = [0] * 30
    assert team_switches(frames_for(labels), labels) == []


def test_lasting_kit_change_is_an_id_switch():
    labels = [0] * 12 + [1] * 12
    assert team_switches(frames_for(labels), labels) == [36]


def test_brief_occlusion_noise_does_not_split():
    labels = [0] * 12 + [1] * 3 + [0] * 12
    assert team_switches(frames_for(labels), labels) == []


def test_noisy_start_does_not_split():
    labels = [1, 1] + [0] * 20
    assert team_switches(frames_for(labels), labels) == []


def test_switch_and_switch_back_gives_two_splits():
    labels = [0] * 10 + [1] * 10 + [0] * 10
    assert team_switches(frames_for(labels), labels) == [30, 60]


def test_kits_far_from_both_teams_are_outliers():
    centres = np.array([[90.0, 140, 145], [60.0, 130, 122]])  # red kit, blue kit
    tracks = {1: [(85, 139, 146)] * 5, 2: [(92, 141, 144)] * 5, 3: [(58, 131, 121)] * 5,
              4: [(62, 129, 123)] * 5,
              9: [(200, 100, 190)] * 5}  # neon-green bib on the touchline
    assert kit_outliers(tracks, centres) == {9}


def test_no_outliers_when_every_kit_matches_a_team():
    centres = np.array([[90.0, 140, 145], [60.0, 130, 122]])
    tracks = {1: [(88, 140, 145)] * 3, 2: [(61, 130, 121)] * 3}
    assert kit_outliers(tracks, centres) == set()

def test_lasting_kit_change_without_any_occlusion_is_not_a_swap():
    labels = [0] * 12 + [1] * 12  # e.g. walked from sun into shadow, nobody nearby
    assert team_switches(frames_for(labels), labels, occluded=[False] * 24) == []


def test_kit_change_right_after_an_occlusion_is_a_swap():
    labels = [0] * 12 + [1] * 12
    occluded = [False] * 10 + [True] * 3 + [False] * 11  # boxes overlapped around the flip
    assert team_switches(frames_for(labels), labels, occluded=occluded) == [36]