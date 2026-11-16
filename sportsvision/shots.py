"""Basketball shot state machine over per-frame ball, rim and shooting-pose cues.

attempt: ball above the rim within 2 rim widths, or ball-in-basket at the rim; backdated to
a preceding shooting pose (a pose alone never counts). made: the ball crosses rim height
downwards inside the rim span, or ball-in-basket on 2+ frames. missed: crossing outside the
rim or timeout. A cooldown stops one rebound counting twice."""
from dataclasses import dataclass

Point = tuple[float, float]
Box = tuple[float, float, float, float]


@dataclass
class ShotEvent:
    frame: int
    kind: str  # "attempt" | "made" | "missed"


@dataclass
class Shot:
    start_frame: int
    end_frame: int
    outcome: str  # "made" | "missed"


class ShotDetector:
    def __init__(self, fps: float, zone_width: float = 2.0, timeout_s: float = 2.5,
                 cooldown_s: float = 1.0, rim_margin: float = 0.15, debounce: int = 3):
        self.zone_width = zone_width          # shot zone half-width, in rim widths
        self.rim_margin = rim_margin          # tolerance on the rim span, in rim widths
        self.timeout = int(timeout_s * fps)
        self.cooldown = int(cooldown_s * fps)
        self.debounce = debounce              # consecutive shooting-pose frames to trust it
        self.shots: list[Shot] = []
        self._start: int | None = None        # reported start (maybe backdated to release)
        self._t0 = 0                          # frame the attempt was confirmed (timeout clock)
        self._above: Point | None = None      # last ball position above the rim
        self._in_basket = 0
        self._last_end = -10**9
        self._run = 0                         # consecutive frames with a shooting pose
        self._release: int | None = None      # first frame of the latest shooting pose

    def update(self, frame: int, ball: Point | None, rim: Box | None,
               ball_in_basket: bool = False, shooting: bool = False) -> list[ShotEvent]:
        self._run = self._run + 1 if shooting else 0
        if self._run == self.debounce:
            self._release = frame - self.debounce + 1
        # ball-in-basket only counts at the rim: logos of a ball on ad boards fire it too.
        ball_in_basket = ball_in_basket and self._in_net(ball, rim)
        if self._start is None:
            return self._look_for_attempt(frame, ball, rim, ball_in_basket)
        if ball_in_basket:
            self._in_basket += 1
            if self._in_basket >= 2:
                return self._resolve(frame, "made")
        if frame - self._t0 > self.timeout:
            return self._resolve(frame, "missed")
        if ball is None or rim is None:
            return []
        x1, y1, x2, y2 = rim
        rim_y = (y1 + y2) / 2
        if ball[1] < rim_y:
            self._above = ball
            return []
        if self._above is None:  # attempt came from ball-in-basket; no arc seen yet
            return []
        # Ball has dropped below rim height: interpolate where it crossed.
        # ponytail: 2-D rim-line test, a front-rim bounce-out can look like a make from one
        # camera; add a fall-speed check (rim = 45 cm pixel scale) if makes are over-counted.
        ax, ay = self._above
        cross_x = ax + (rim_y - ay) / (ball[1] - ay) * (ball[0] - ax)
        margin = self.rim_margin * (x2 - x1)
        made = x1 - margin <= cross_x <= x2 + margin
        return self._resolve(frame, "made" if made else "missed")

    def _look_for_attempt(self, frame, ball, rim, ball_in_basket) -> list[ShotEvent]:
        if frame - self._last_end < self.cooldown:
            return []
        in_zone = False
        if ball is not None and rim is not None:
            x1, y1, x2, y2 = rim
            in_zone = abs(ball[0] - (x1 + x2) / 2) <= self.zone_width * (x2 - x1) and ball[1] < y1
        if not (in_zone or ball_in_basket):
            return []
        recent_release = self._release is not None and frame - self._release <= self.timeout
        self._start = self._release if recent_release else frame
        self._t0, self._release = frame, None
        self._above = ball if in_zone else None
        self._in_basket = int(ball_in_basket)
        return [ShotEvent(self._start, "attempt")]

    @staticmethod
    def _in_net(ball: Point | None, rim: Box | None) -> bool:
        """Ball centre within one rim width sideways and in the net's depth below the rim."""
        if ball is None or rim is None:
            return False
        x1, y1, x2, y2 = rim
        w = x2 - x1
        dy = ball[1] - (y1 + y2) / 2
        return abs(ball[0] - (x1 + x2) / 2) <= w and -0.5 * w <= dy <= 1.5 * w

    def _resolve(self, frame: int, outcome: str) -> list[ShotEvent]:
        self.shots.append(Shot(self._start, frame, outcome))
        self._start, self._above, self._last_end = None, None, frame
        return [ShotEvent(frame, outcome)]
