"""Depth-based obstacle avoidance by follow-the-gap.

Replaces the blind open-loop go-around. The old design waited for the obstacle
to enter the D435i's minimum-range blind zone (OBSTACLE_TRIGGER_ON_DROPOUT),
threw away the depth it had, and then dead-reckoned a box around it -- on a
robot with no encoders and (on the USB-2 link) no IMU, so every leg ran to its
timeout. This steers on live depth instead, while the obstacle is still 300-500mm
out and clearly measurable.

Method is the F1TENTH "follow the gap" algorithm (f1tenth_lab4), fed from the
depth image the way ArduPilot's d4xx_to_mavlink script does it: collapse a band
of rows to one range per column, find the closest return, mask a safety bubble
around it, then steer at the middle of the widest remaining free run.

Two adaptations matter here, both called out as the usual failure modes when
swapping a LiDAR for a depth camera:

  * Invalid (zero) depth must NOT read as free space, or the robot steers
    confidently into whatever it could not measure. Zeros are carried as
    unknown and only cleared when the column has real evidence of floor.
  * The usable row band is bounded by geometry, not preference. At 60mm mount
    height and 25 deg down-tilt, rows below ~40% of the frame image ground
    closer than the sensor's ~105mm minimum range, so their depth is noise
    that reads nearer-than-floor and paints phantom obstacles.
"""
import numpy as np

from config import (
    GROUND_DEPTH_TOLERANCE_MM, OBSTACLE_BAND_MAX_RANGE_MM,
    OBSTACLE_BAND_IGNORE_BELOW_FRAC, GAP_BUBBLE_COLS, GAP_MIN_WIDTH_COLS,
    GAP_COLUMN_MIN_HITS, GAP_UNKNOWN_MAX_FRAC,
)


def depth_to_scan(depth_mm, expected_map):
    """Collapse the depth image to one obstacle range per column.

    Returns (ranges, blocked) where ranges[x] is the distance in mm to the
    nearest thing in column x that is closer than the floor should be
    (OBSTACLE_BAND_MAX_RANGE_MM if the column is clear), and blocked[x] marks
    columns that are either occupied or too unmeasurable to trust.
    """
    h, w = depth_mm.shape[:2]
    top = 0
    bottom = int(h * OBSTACLE_BAND_IGNORE_BELOW_FRAC)
    band = depth_mm[top:bottom, :]
    expected = expected_map[top:bottom, :]

    valid = band > 0
    # Nearer than the floor by a clear margin, and near enough to be on our
    # tile rather than a wall across the room.
    obstacle = valid & ((expected - band) > GROUND_DEPTH_TOLERANCE_MM) \
        & (band <= OBSTACLE_BAND_MAX_RANGE_MM)

    hits = obstacle.sum(axis=0)
    ranges = np.full(w, float(OBSTACLE_BAND_MAX_RANGE_MM), dtype=np.float32)
    occupied = hits >= GAP_COLUMN_MIN_HITS
    if occupied.any():
        masked = np.where(obstacle, band, np.inf)
        ranges[occupied] = masked[:, occupied].min(axis=0)

    # A column with almost no valid depth cannot be called clear. This is the
    # blind-zone case: an obstacle close enough to stop returning depth reads
    # as a hole, and treating holes as free space is how gap-following drives
    # into things it cannot see.
    unknown_frac = 1.0 - valid.mean(axis=0)
    unmeasurable = unknown_frac > GAP_UNKNOWN_MAX_FRAC

    blocked = occupied | unmeasurable
    ranges[unmeasurable & ~occupied] = 0.0
    return ranges, blocked


def find_gap(blocked, closest_col=None):
    """Widest run of free columns. Returns (start, end, centre) or None.

    Mirrors the lab4 sequence: bubble out the closest return first so the
    chosen gap cannot graze the obstacle, then take the longest survivor.
    """
    free = ~blocked.copy()
    if closest_col is not None:
        lo = max(0, closest_col - GAP_BUBBLE_COLS)
        hi = min(len(free), closest_col + GAP_BUBBLE_COLS + 1)
        free[lo:hi] = False

    best_start = best_len = run_start = run_len = 0
    for x in range(len(free)):
        if free[x]:
            if run_len == 0:
                run_start = x
            run_len += 1
            if run_len > best_len:
                best_len, best_start = run_len, run_start
        else:
            run_len = 0

    if best_len < GAP_MIN_WIDTH_COLS:
        return None
    return best_start, best_start + best_len, best_start + best_len // 2


def gap_steer(depth_mm, expected_map):
    """One avoidance decision from one depth frame.

    Returns a dict describing what the depth sees and where to aim:
      obstacle_mm  range to the nearest obstacle, None if the view is clear
      steer        -1..+1 offset to the middle of the widest gap, None if none
      gap_width    width of that gap as a fraction of the frame
    """
    result = {"obstacle_mm": None, "steer": None, "gap_width": 0.0,
              "blocked_frac": 0.0}
    if depth_mm is None or expected_map is None:
        return result

    ranges, blocked = depth_to_scan(depth_mm, expected_map)
    w = len(ranges)
    result["blocked_frac"] = float(blocked.mean())
    if not blocked.any():
        return result

    closest_col = int(np.argmin(np.where(blocked, ranges, np.inf)))
    result["obstacle_mm"] = float(ranges[closest_col])

    gap = find_gap(blocked, closest_col)
    if gap is None:
        return result
    start, end, centre = gap
    result["steer"] = float((centre - w / 2.0) / (w / 2.0))
    result["gap_width"] = float(end - start) / float(w)
    return result
