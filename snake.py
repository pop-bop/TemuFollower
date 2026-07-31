"""Snake line tracer, ported from Arne Baeyens' RCJ 2014 line follower.

https://github.com/abaeyens/image-processing (Apache-2.0). The original walks
the line with circular pixel scans: find the line on a bottom scanline, then
repeatedly scan a small circle around the last hit -- each intersection is the
next line point, and the chain of points gives the line's local angle. His own
2019 note advises against porting the pixel loops to Python; this port keeps
the snake idea but replaces each circle scan with a windowed centroid on the
already-thresholded line mask, which is one cv2.moments call per step.

The chain is what the flat ROIs cannot give: a marker's side must be judged
against the LINE's local direction, not the image x axis -- a green square
that reads "left of centroid" while the robot enters a corner at 30 degrees
may well be on the line's right. The cross product against the traced segment
gets this right at any approach angle.
"""
import cv2
import numpy as np

from config import (
    SNAKE_STEP_PX, SNAKE_WINDOW_PX, SNAKE_MAX_STEPS, SNAKE_MIN_PIXELS,
)


def trace_line(black_mask, start_xy=None):
    """Walk the line mask bottom-up. Returns a list of (x, y) line points.

    Mirrors the original's zero scan + chained circle scans: the first point
    comes from the widest window at the bottom row band (the "scanline"), each
    later point from a window centred one SNAKE_STEP_PX above the previous
    point ("scancircle"). Stops when a window holds too few line pixels --
    end of the line, a gap, or the frame edge.
    """
    h, w = black_mask.shape[:2]
    points = []

    # Zero scan: find the entry point along the bottom band.
    if start_xy is None:
        band = black_mask[h - SNAKE_STEP_PX:h, :]
        m = cv2.moments(band, binaryImage=True)
        if m["m00"] < SNAKE_MIN_PIXELS:
            return points
        x = m["m10"] / m["m00"]
        y = h - SNAKE_STEP_PX / 2.0
    else:
        x, y = start_xy
    points.append((x, y))

    # Chained scans: window centred a step along the current heading.
    dx, dy = 0.0, -float(SNAKE_STEP_PX)
    half = SNAKE_WINDOW_PX // 2
    for _ in range(SNAKE_MAX_STEPS):
        cx, cy = points[-1][0] + dx, points[-1][1] + dy
        x0 = int(max(0, cx - half)); x1 = int(min(w, cx + half))
        y0 = int(max(0, cy - half)); y1 = int(min(h, cy + half))
        if x1 - x0 < 2 or y1 - y0 < 2:
            break
        win = black_mask[y0:y1, x0:x1]
        m = cv2.moments(win, binaryImage=True)
        if m["m00"] < SNAKE_MIN_PIXELS:
            break
        nx = x0 + m["m10"] / m["m00"]
        ny = y0 + m["m01"] / m["m00"]
        # Heading update, clamped like the original's +/-45 deg look angle so
        # one noisy window cannot fold the snake back onto itself.
        ndx, ndy = nx - points[-1][0], ny - points[-1][1]
        norm = (ndx * ndx + ndy * ndy) ** 0.5
        if norm < 1.0 or ndy > 0:
            break
        scale = SNAKE_STEP_PX / norm
        dx, dy = ndx * scale, ndy * scale
        points.append((nx, ny))

    return points


def side_of_line(points, px, py):
    """-1 if (px, py) lies left of the traced line, +1 if right, 0 if unknown.

    Left/right follow the LINE's travel direction (bottom of frame upward),
    which is the robot's own left/right. Uses the segment whose midpoint is
    nearest the query point, so a marker beside a corner is judged against
    the corner's local direction rather than a distant straight stretch.
    """
    if len(points) < 2:
        return 0
    best, best_d2 = None, None
    for (ax, ay), (bx, by) in zip(points[:-1], points[1:]):
        mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
        d2 = (px - mx) ** 2 + (py - my) ** 2
        if best_d2 is None or d2 < best_d2:
            best, best_d2 = ((ax, ay), (bx, by)), d2
    (ax, ay), (bx, by) = best
    # Image y grows downward, so the sign of the 2D cross product flips
    # relative to maths convention: positive cross = marker on the RIGHT.
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if abs(cross) < 1e-6:
        return 0
    return 1 if cross > 0 else -1


def classify_green_markers(black_mask, green_mask, min_area, max_ahead_px=None):
    """Trace the line, then place each green blob on its left or right.

    Returns (green_left, green_right, line_points). Both True = dead end
    (RescueLine 3.6.4). Blobs are only judged when the trace produced at
    least one segment; a failed trace returns (False, False, points) so a
    frame with no traceable line can never latch a turn.

    max_ahead_px enforces RescueLine 3.6.6 -- markers are placed JUST BEFORE
    the intersection, so only the ones the robot has yet to reach are its
    own. A marker beyond the intersection belongs to a robot approaching from
    another branch and must be ignored, or the robot turns on someone else's
    instruction. Distance is measured ALONG the traced line rather than in
    image y, so the rule still holds when the intersection is approached at
    an angle.
    """
    points = trace_line(black_mask)
    if len(points) < 2:
        return False, False, points

    # Arc length from the bottom of the trace (the robot end) to each point.
    arc = [0.0]
    for (ax, ay), (bx, by) in zip(points[:-1], points[1:]):
        arc.append(arc[-1] + ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5)

    left = right = False
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        m = cv2.moments(c)
        if m["m00"] <= 0:
            continue
        px, py = m["m10"] / m["m00"], m["m01"] / m["m00"]
        if max_ahead_px is not None:
            nearest = min(range(len(points)),
                          key=lambda i: (px - points[i][0]) ** 2
                          + (py - points[i][1]) ** 2)
            if arc[nearest] > max_ahead_px:
                continue
        side = side_of_line(points, px, py)
        if side < 0:
            left = True
        elif side > 0:
            right = True
    return left, right, points
