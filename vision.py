import cv2
import numpy as np

from config import (
    ROI_Y_START_RATIO, ROI_Y_END_RATIO, ROI_X_START_RATIO, ROI_X_END_RATIO,
    LOOKAHEAD_ROI_Y_START_RATIO, LOOKAHEAD_ROI_Y_END_RATIO,
    LOOKAHEAD_ROI_X_START_RATIO, LOOKAHEAD_ROI_X_END_RATIO,
    FAR_ROI_Y_START_RATIO, FAR_ROI_Y_END_RATIO,
    FAR_ROI_X_START_RATIO, FAR_ROI_X_END_RATIO,
    WIDE_ROI_Y_START_RATIO, WIDE_ROI_X_START_RATIO, WIDE_ROI_X_END_RATIO,
    BLACK_THRESHOLD, MIN_LINE_AREA, LINE_CONFIDENCE_FULL_AREA,
    GREEN_DIFF_THRESHOLD, RED_DIFF_THRESHOLD,
    MIN_MARKER_AREA, INTERSECTION_MIN_AREA, INTERSECTION_MIN_CONTOURS,
    WARP_MATRIX, GROUND_DEPTH_TOLERANCE_MM, OBSTACLE_DETECTION_ENABLED,
    OBSTACLE_BAND_ENABLED,
    OBSTACLE_BAND_MIN_WIDTH_RATIO, OBSTACLE_BAND_MAX_RANGE_MM,
    OBSTACLE_BAND_IGNORE_BELOW_FRAC,
)
from utils import clamp
from snake import classify_green_markers
from config import SNAKE_MARKER_MAX_AHEAD_PX

def get_warp_matrix():
    return np.array(WARP_MATRIX, dtype=np.float32)

def warp_frame(frame, M):
    # WARP_MATRIX is the identity by default, and warpPerspective on a 640x480
    # frame is not free -- it was resampling every frame (twice, when depth is on)
    # just to produce a pixel-for-pixel copy. Skip it unless a real homography is
    # configured.
    if M is None or np.array_equal(M, np.eye(3, dtype=np.float32)):
        return frame
    h, w = frame.shape[:2]
    return cv2.warpPerspective(frame, M, (w, h))

def get_expected_depth_map(w, h, tilt_deg, height_mm, vfov_deg=42.0):
    expected = np.zeros((h, w), dtype=np.float32)
    fov_y = vfov_deg * np.pi / 180.0
    tilt_rad = tilt_deg * np.pi / 180.0
    # Vectorised over rows; every column in a row shares a depression angle, which
    # ignores horizontal FOV and so is only an approximation near the frame edges.
    ny = (np.arange(h, dtype=np.float32) - h / 2.0) / (h / 2.0)
    # Rows lower in the image (ny > 0) look further DOWN, so they strike the
    # ground nearer the robot. The depression angle therefore grows with ny.
    angle = np.maximum(tilt_rad + ny * (fov_y / 2.0), 0.05)
    expected[:] = (height_mm / np.sin(angle))[:, None]
    return expected

def get_obstacle_mask(depth_frame, depth_scale, expected_map):
    if not OBSTACLE_DETECTION_ENABLED:
        return None
    depth_mm = depth_frame * (depth_scale * 1000.0)
    obs = (expected_map - depth_mm) > GROUND_DEPTH_TOLERANCE_MM
    obs = obs & (depth_mm > 0)
    return obs.astype(np.uint8) * 255


def detect_obstacle_band(depth_frame, depth_scale, expected_map=None):
    """Find an upright obstacle as a wide column run reading nearer than floor.

    Geometry note, because it is counter-intuitive: a downward ray reaches the
    floor before an obstacle whenever the floor is nearer, so an obstacle does
    NOT fill its columns. It is anchored at the TOP of the frame and grows
    downward as range closes -- about 7% of frame height at 500mm, 42% at 150mm.
    The cue is therefore "measured depth well nearer than the expected GROUND
    depth for that row", counted per column. No colour is used: red marks the
    goal tile and the dead-victim point, not obstacles.

    Returns a dict with:
      found            band believed present this frame
      range_mm         nearest depth within the band
      center_offset    band centre as -1..+1 across the frame
      width_ratio      band width as a fraction of frame width
      fill_ratio       how much of the band's height reads as obstacle; grows
                       as the robot closes in
      left_clearance_mm / right_clearance_mm
                       median depth either side of the band, for choosing a
                       side to pass on
      any_valid_depth  whether the frame carried usable depth at all, so the
                       caller can tell "no obstacle" from "sensor blind"
    """
    result = {
        "found": False, "range_mm": None, "center_offset": 0.0,
        "width_ratio": 0.0, "fill_ratio": 0.0, "left_clearance_mm": None,
        "right_clearance_mm": None, "any_valid_depth": False,
    }
    if not OBSTACLE_BAND_ENABLED or depth_frame is None or expected_map is None:
        return result

    depth_mm = depth_frame.astype(np.float32) * (depth_scale * 1000.0)
    h, w = depth_mm.shape[:2]
    # The bottom rows image floor inside the sensor's blind zone, where USB-2
    # depth reads garbage non-zero values that beat the expected-floor map and
    # paint phantom bands. An upright obstacle is anchored at the TOP of the
    # frame (all rays point down), so dropping the bottom costs nothing.
    depth_mm[int(h * OBSTACLE_BAND_IGNORE_BELOW_FRAC):, :] = 0.0
    valid = depth_mm > 0
    result["any_valid_depth"] = bool(valid.any())
    if not result["any_valid_depth"]:
        return result

    # Nearer than the floor would be by a clear margin, and near enough to be
    # an obstacle on our tile rather than a wall across the room.
    nearer = valid & ((expected_map - depth_mm) > GROUND_DEPTH_TOLERANCE_MM)
    nearer &= depth_mm <= OBSTACLE_BAND_MAX_RANGE_MM

    col_frac = nearer.mean(axis=0)
    # An obstacle at the far end of the usable range still only fills a few
    # percent of its column, so this floor has to stay low.
    is_obstacle_col = col_frac >= 0.04
    if not is_obstacle_col.any():
        return result

    best_start = best_len = run_start = run_len = 0
    for x in range(w):
        if is_obstacle_col[x]:
            if run_len == 0:
                run_start = x
            run_len += 1
            if run_len > best_len:
                best_len, best_start = run_len, run_start
        else:
            run_len = 0

    if best_len < max(1, int(w * OBSTACLE_BAND_MIN_WIDTH_RATIO)):
        return result

    x0, x1 = best_start, best_start + best_len
    band_hits = depth_mm[:, x0:x1][nearer[:, x0:x1]]
    if band_hits.size == 0:
        return result

    result["found"] = True
    result["range_mm"] = float(np.percentile(band_hits, 10))
    result["width_ratio"] = best_len / float(w)
    result["fill_ratio"] = float(nearer[:, x0:x1].mean())
    center_px = (x0 + x1) / 2.0
    result["center_offset"] = float((center_px - w / 2.0) / (w / 2.0))

    # Free space either side, used to pick a passing side. Median depth is
    # useless here -- open floor reads the same on both sides -- so measure the
    # nearest OBSTRUCTION instead: the closest thing on that side reading nearer
    # than the floor should be. No obstruction leaves the side wide open.
    for key, sl in (("left_clearance_mm", np.s_[:, :x0]),
                    ("right_clearance_mm", np.s_[:, x1:])):
        region = depth_mm[sl]
        blocked = nearer[sl]
        if region.size == 0:
            result[key] = 0.0
        elif blocked.any():
            result[key] = float(np.percentile(region[blocked], 10))
        else:
            result[key] = float(OBSTACLE_BAND_MAX_RANGE_MM)

    return result


def find_line_error(frame, obstacle_mask, y_start_ratio, y_end_ratio, x_start_ratio, x_end_ratio):
    h, w = frame.shape[:2]
    y0 = int(h * y_start_ratio)
    y1 = int(h * y_end_ratio)
    x0 = int(w * x_start_ratio)
    x1 = int(w * x_end_ratio)
    roi = frame[y0:y1, x0:x1]
    roi_w = x1 - x0

    if obstacle_mask is not None:
        obs_roi = obstacle_mask[y0:y1, x0:x1]
    else:
        obs_roi = np.zeros((y1-y0, roi_w), dtype=np.uint8)

    b, g, r = cv2.split(roi)
    _, gr = cv2.threshold(cv2.subtract(g, r), GREEN_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    _, gb = cv2.threshold(cv2.subtract(g, b), GREEN_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    green_mask = cv2.bitwise_and(gr, gb)
    
    _, rg = cv2.threshold(cv2.subtract(r, g), RED_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    _, rb = cv2.threshold(cv2.subtract(r, b), RED_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    red_mask = cv2.bitwise_and(rg, rb)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Global threshold, not adaptiveThreshold. Adaptive is relative to the local
    # neighbourhood mean, so it has no concept of "no line here": on blank floor
    # it thresholds sensor noise and reports a 26000px contour with a confident
    # error. Measured on the blank-floor frames from the 2026-07-30 run, where
    # the robot drove at turn=+0.80 against an empty mat. A global cut says LINE
    # LOST on those same frames, which is the correct answer.
    _, black_mask = cv2.threshold(blur, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    black_mask[green_mask > 0] = 0
    black_mask[red_mask > 0] = 0
    black_mask[obs_roi > 0] = 0 # Reject blocks

    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    debug_info = {
        "roi_bounds": (x0, y0, x1, y1),
        "mask": black_mask,
        "line_point": None,
        "roi_frame": roi,
        "red_marker": False,
        "green_marker": cv2.countNonZero(green_mask) >= MIN_MARKER_AREA,
        "green_left": False,
        "green_right": False,
        "line_area": 0.0,
        "line_confidence": 0.0,
        "obstacle_detected": cv2.countNonZero(obs_roi) > 0
    }

    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, debug_info

    largest = max(contours, key=cv2.contourArea)
    line_area = cv2.contourArea(largest)
    debug_info["line_area"] = line_area
    # Scale against a healthy line, not MIN_LINE_AREA*8. That reference was set
    # when MIN_LINE_AREA was 45 at 320x240 (ref 360, comparable to a real line).
    # After the 640x480 rescale to 180 the reference became 1440 while a plain
    # straight line measures ~9700px, so this pinned to 1.0 on every frame and
    # silently disabled LOW_CONFIDENCE_SPEED_SCALE, ADAPTIVE_KP_CONFIDENCE_DROP,
    # LOOKAHEAD_CONFIDENCE_MIN and FAR_CONFIDENCE_MIN.
    debug_info["line_confidence"] = clamp(
        line_area / max(float(LINE_CONFIDENCE_FULL_AREA), 1.0), 0.0, 1.0)
    if line_area < MIN_LINE_AREA:
        return None, debug_info

    # A line is a stripe within the ROI, so it cannot fill the ROI. When the
    # frame goes dark -- auto-exposure still converging, or a light change --
    # EVERY pixel falls under the threshold and the "line" becomes one blob
    # covering the whole ROI, whose centroid is the ROI centre. That reads as a
    # perfectly centred line (error 0.00) at maximum confidence, so the robot
    # drives straight on no evidence. Measured at 51750px of a 52416px ROI.
    if line_area > 0.60 * black_mask.size:
        debug_info["line_confidence"] = 0.0
        return None, debug_info

    near_line_zone = np.zeros(black_mask.shape, dtype=np.uint8)
    cv2.drawContours(near_line_zone, [largest], -1, 255, thickness=cv2.FILLED)
    near_line_zone = cv2.dilate(near_line_zone, np.ones((15, 15), np.uint8))

    red_hit = cv2.bitwise_and(near_line_zone, red_mask)
    red_marker = cv2.countNonZero(red_hit) >= MIN_MARKER_AREA
    if red_marker:
        red_hit_obs = cv2.bitwise_and(red_hit, obs_roi)
        if cv2.countNonZero(red_hit_obs) > (MIN_MARKER_AREA / 2):
            red_marker = False # It's a block, not a flat line marker
            
    debug_info["red_marker"] = red_marker

    M = cv2.moments(largest)
    if M["m00"] <= 0:
        return None, debug_info

    cx_roi = int(M["m10"] / M["m00"])
    cy_roi = int(M["m01"] / M["m00"])

    # Which side of the LINE each green marker sits on (RescueLine 3.6: left
    # marker = turn left, right = turn right, both = dead end / turn around).
    # The snake trace gives the line's LOCAL direction, so a marker beside a
    # corner entered at an angle is still judged correctly -- against the
    # line, not the image axis. Only blobs near the line count: green
    # evacuation triangles and stray off-tile objects must not latch a turn.
    green_near = cv2.bitwise_and(near_line_zone, green_mask)
    if cv2.countNonZero(green_near) >= MIN_MARKER_AREA:
        gl, gr_side, _ = classify_green_markers(
            black_mask, green_near, MIN_MARKER_AREA / 2,
            max_ahead_px=SNAKE_MARKER_MAX_AHEAD_PX)
        debug_info["green_left"] = gl
        debug_info["green_right"] = gr_side
    cx_global = cx_roi + x0
    cy_global = cy_roi + y0
    center_x_global = x0 + roi_w // 2
    error = (cx_global - center_x_global) / max(1.0, roi_w / 2.0)

    # Obstacle avoidance: steer away from blocks. The offset scales with how far
    # off-centre the obstacle is; a binary +/-0.6 flapped sign every frame whenever
    # the obstacle centroid sat near the ROI midline.
    obs_pixels = cv2.countNonZero(obs_roi)
    if obs_pixels > 0:
        obs_m = cv2.moments(obs_roi)
        if obs_m["m00"] > 0:
            obs_cx = obs_m["m10"] / obs_m["m00"]
            obs_offset = (obs_cx - roi_w / 2.0) / max(1.0, roi_w / 2.0)
            error -= 0.6 * clamp(obs_offset, -1.0, 1.0)

    debug_info["line_point"] = (cx_global, cy_global)
    return clamp(error, -1.0, 1.0), debug_info


def find_line_error_normal(frame, obstacle_mask=None):
    return find_line_error(
        frame, obstacle_mask, ROI_Y_START_RATIO, ROI_Y_END_RATIO,
        ROI_X_START_RATIO, ROI_X_END_RATIO,
    )


def find_line_error_lookahead(frame, obstacle_mask=None):
    return find_line_error(
        frame, obstacle_mask, LOOKAHEAD_ROI_Y_START_RATIO, LOOKAHEAD_ROI_Y_END_RATIO,
        LOOKAHEAD_ROI_X_START_RATIO, LOOKAHEAD_ROI_X_END_RATIO,
    )


def find_line_error_far(frame, obstacle_mask=None):
    return find_line_error(
        frame, obstacle_mask, FAR_ROI_Y_START_RATIO, FAR_ROI_Y_END_RATIO,
        FAR_ROI_X_START_RATIO, FAR_ROI_X_END_RATIO,
    )


def find_line_error_wide(frame, obstacle_mask=None):
    return find_line_error(
        frame, obstacle_mask, WIDE_ROI_Y_START_RATIO, 1.0,
        WIDE_ROI_X_START_RATIO, WIDE_ROI_X_END_RATIO,
    )


def detect_intersection(frame, obstacle_mask, y_start_ratio, y_end_ratio, x_start_ratio, x_end_ratio):
    h, w = frame.shape[:2]
    y0 = int(h * y_start_ratio)
    y1 = int(h * y_end_ratio)
    x0 = int(w * x_start_ratio)
    x1 = int(w * x_end_ratio)
    roi = frame[y0:y1, x0:x1]
    roi_w = x1 - x0
    roi_center_x = x0 + roi_w // 2

    if obstacle_mask is not None:
        obs_roi = obstacle_mask[y0:y1, x0:x1]
    else:
        obs_roi = np.zeros((y1-y0, roi_w), dtype=np.uint8)

    b, g, r = cv2.split(roi)
    _, gr = cv2.threshold(cv2.subtract(g, r), GREEN_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    _, gb = cv2.threshold(cv2.subtract(g, b), GREEN_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    green_mask = cv2.bitwise_and(gr, gb)
    _, rg = cv2.threshold(cv2.subtract(r, g), RED_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    _, rb = cv2.threshold(cv2.subtract(r, b), RED_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    red_mask = cv2.bitwise_and(rg, rb)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    mask[green_mask > 0] = 0
    mask[red_mask > 0] = 0
    mask[obs_roi > 0] = 0

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant = [c for c in contours if cv2.contourArea(c) >= INTERSECTION_MIN_AREA]

    if len(significant) < INTERSECTION_MIN_CONTOURS:
        return None

    contour_info = []
    for c in significant:
        M = cv2.moments(c)
        if M["m00"] <= 0:
            continue
        cx = int(M["m10"] / M["m00"]) + x0
        cy = int(M["m01"] / M["m00"]) + y0
        area = cv2.contourArea(c)
        contour_info.append((cx, cy, area))

    if len(contour_info) < 2:
        return None

    contour_info.sort(key=lambda info: info[0])
    chosen_idx = max(range(len(contour_info)), key=lambda i: contour_info[i][2])
    branches = [{"cx": cx, "cy": cy, "taken": False} for cx, cy, _ in contour_info]

    return {
        "is_intersection": True,
        "branch_count": len(branches),
        "branches": branches,
        "chosen_branch_idx": chosen_idx,
        "roi_center_x": roi_center_x,
        "centroid_x": sum(b["cx"] for b in branches) // len(branches),
        "centroid_y": sum(b["cy"] for b in branches) // len(branches),
    }


def detect_intersection_normal(frame, obstacle_mask=None):
    return detect_intersection(
        frame, obstacle_mask, ROI_Y_START_RATIO, ROI_Y_END_RATIO,
        ROI_X_START_RATIO, ROI_X_END_RATIO,
    )


def print_calibration_info(frame):
    h, w = frame.shape[:2]
    x0 = int(w * ROI_X_START_RATIO)
    x1 = int(w * ROI_X_END_RATIO)
    center_x_global = x0 + (x1 - x0) // 2
    print("Calibration check:")
    print(f"  frame size: {w}x{h}")
    print(f"  ROI x range: {x0} to {x1}")
    print(f"  ROI center x (should match robot centerline): {center_x_global}")
    print(f"  frame center x: {w // 2}")
    if abs(center_x_global - w // 2) > 2:
        print("  NOTE: ROI center is not the same as frame center. This is expected")
        print("  only if the camera is intentionally offset. Otherwise consider")
        print("  adjusting ROI_X_START_RATIO / ROI_X_END_RATIO or the camera mount.")
