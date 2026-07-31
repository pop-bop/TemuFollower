"""Green intersection markers as an explicit decision graph (RescueLine 3.6).

The previous version scattered marker handling across find_line_error (which
ran a snake trace on EVERY ROI of EVERY frame, inside the steering hot path)
and a pile of counters in the main loop. Two problems: the cost landed on the
control loop, and the actual decision was spread over enough places that
"which branch do I take" had no single answer to read.

Everything here is pure: it takes a frame and returns a decision. Nothing
touches motors, so it can be tested exhaustively off-robot.

THE DECISION GRAPH
==================

                        +---------------------+
                        |  green blobs near   |
                        |  the line, in the   |
                        |  marker band?       |
                        +----------+----------+
                                   |
                    no             |            yes
              +--------------------+--------------------+
              |                                         |
              v                                         v
      +---------------+                    +-------------------------+
      | GO_STRAIGHT   |                    | classify each blob as    |
      | (rule 3.6.3)  |                    | LEFT or RIGHT of the     |
      +---------------+                    | line's local direction   |
                                           +------------+-------------+
                                                        |
                        +-------------------------------+---------------+
                        |                   |                           |
                   left only           right only                  both sides
                        |                   |                           |
                        v                   v                           v
                 +------------+      +------------+            +----------------+
                 | TURN_LEFT  |      | TURN_RIGHT |            | U_TURN         |
                 | (3.6.2)    |      | (3.6.2)    |            | dead end 3.6.4 |
                 +------------+      +------------+            +----------------+

Confirmation: a decision is only returned after the SAME verdict is seen on
MARKER_CONFIRM_FRAMES consecutive frames. A single frame of green is a
reflection or a compression artefact; a real 25mm marker is in view for many
frames as the robot approaches it.

Rule 3.6.6 ("markers are placed just before the intersection") is enforced by
the marker band: only blobs in the lower part of the frame -- ground the robot
is about to reach -- can vote. A marker on the far side of the intersection
belongs to a robot approaching from another branch.
"""
import cv2
import numpy as np

from config import (
    GREEN_DIFF_THRESHOLD, MIN_MARKER_AREA, BLACK_THRESHOLD,
    MARKER_BAND_Y_START, MARKER_BAND_Y_END,
    MARKER_MAX_LINE_DIST_PX, MARKER_CONFIRM_FRAMES,
)

GO_STRAIGHT = "GO_STRAIGHT"
TURN_LEFT = "TURN_LEFT"
TURN_RIGHT = "TURN_RIGHT"
U_TURN = "U_TURN"


def green_mask_of(bgr):
    """Green where it beats both red and blue by a margin.

    Channel differencing rather than HSV: it is insensitive to the brightness
    swings across the mat, and it cannot be fooled by the desaturated grey the
    black line takes on under strong light the way an H-range can.
    """
    b, g, r = cv2.split(bgr)
    _, gr = cv2.threshold(cv2.subtract(g, r), GREEN_DIFF_THRESHOLD, 255,
                          cv2.THRESH_BINARY)
    _, gb = cv2.threshold(cv2.subtract(g, b), GREEN_DIFF_THRESHOLD, 255,
                          cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(gr, gb)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def line_mask_of(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def line_axis(line_mask, y_lo, y_hi):
    """Fit the line's local direction over a row band.

    Returns (x_at_bottom, dx_per_row) or None. The direction matters: the side
    a marker sits on must be measured against where the LINE is going, not
    against the image's x axis. Entering a corner at 30 degrees, a marker that
    is plainly on the line's right can sit left of the frame centre, and an
    image-axis test then turns the wrong way.

    Fitted by centroid-per-row least squares, which is stable on a 25mm stripe
    and costs a few hundred microseconds -- no iterative tracing.
    """
    rows, xs = [], []
    for y in range(y_lo, y_hi, 2):
        row = line_mask[y]
        hit = np.flatnonzero(row)
        if hit.size:
            rows.append(y)
            xs.append(hit.mean())
    if len(rows) < 4:
        return None
    rows = np.asarray(rows, dtype=np.float32)
    xs = np.asarray(xs, dtype=np.float32)
    # x = a*y + b, so a is dx per row travelled up the image.
    a, b = np.polyfit(rows, xs, 1)
    return float(a * (y_hi - 1) + b), float(a)


def classify(frame):
    """One frame in, one raw verdict out. No history, no side effects.

    Returns (verdict, info). Verdict is one of the module constants.
    """
    h, w = frame.shape[:2]
    y_lo = int(h * MARKER_BAND_Y_START)
    y_hi = int(h * MARKER_BAND_Y_END)

    lm = line_mask_of(frame)
    gm = green_mask_of(frame)
    info = {"left": False, "right": False, "blobs": [], "axis": None,
            "band": (y_lo, y_hi)}

    axis = line_axis(lm, y_lo, y_hi)
    if axis is None:
        # No line to measure against -- a marker's side is meaningless, and
        # guessing here is how the robot turns on a stray green object.
        return GO_STRAIGHT, info
    x_base, slope = axis
    info["axis"] = axis

    band = gm[y_lo:y_hi, :]
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(band, 8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_MARKER_AREA:
            continue
        cx, cy = centroids[i]
        cy_full = cy + y_lo
        # Where is the line at this blob's row?
        line_x = x_base + slope * (cy_full - (y_hi - 1))
        dist = cx - line_x
        if abs(dist) > MARKER_MAX_LINE_DIST_PX:
            # Too far from the line to be its marker: evacuation-zone
            # triangles and off-tile clutter land here.
            continue
        info["blobs"].append({"cx": float(cx), "cy": float(cy_full),
                              "dist": float(dist),
                              "area": int(stats[i, cv2.CC_STAT_AREA])})
        if dist < 0:
            info["left"] = True
        else:
            info["right"] = True

    if info["left"] and info["right"]:
        return U_TURN, info
    if info["left"]:
        return TURN_LEFT, info
    if info["right"]:
        return TURN_RIGHT, info
    return GO_STRAIGHT, info


class MarkerDecider:
    """Debounces raw per-frame verdicts into a committed decision.

    A verdict must repeat on MARKER_CONFIRM_FRAMES consecutive frames before
    it is acted on. Without this a single reflection latches a turn, which is
    unrecoverable once the robot has left the line.
    """

    def __init__(self):
        self.candidate = GO_STRAIGHT
        self.streak = 0
        self.committed = GO_STRAIGHT

    def reset(self):
        self.candidate = GO_STRAIGHT
        self.streak = 0
        self.committed = GO_STRAIGHT

    def update(self, frame):
        """Feed one frame. Returns (committed_decision, raw_verdict, info)."""
        verdict, info = classify(frame)
        if verdict == self.candidate:
            self.streak += 1
        else:
            self.candidate = verdict
            self.streak = 1
        if self.streak >= MARKER_CONFIRM_FRAMES:
            self.committed = verdict
        return self.committed, verdict, info
