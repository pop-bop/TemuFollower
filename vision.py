import cv2
import numpy as np

from config import (
    ROI_Y_START_RATIO, ROI_Y_END_RATIO, ROI_X_START_RATIO, ROI_X_END_RATIO,
    LOOKAHEAD_ROI_Y_START_RATIO, LOOKAHEAD_ROI_Y_END_RATIO,
    LOOKAHEAD_ROI_X_START_RATIO, LOOKAHEAD_ROI_X_END_RATIO,
    FAR_ROI_Y_START_RATIO, FAR_ROI_Y_END_RATIO,
    FAR_ROI_X_START_RATIO, FAR_ROI_X_END_RATIO,
    WIDE_ROI_Y_START_RATIO, WIDE_ROI_X_START_RATIO, WIDE_ROI_X_END_RATIO,
    BLACK_THRESHOLD, MIN_LINE_AREA, GREEN_DIFF_THRESHOLD, RED_DIFF_THRESHOLD,
    MIN_MARKER_AREA, INTERSECTION_MIN_AREA, INTERSECTION_MIN_CONTOURS,
    WARP_MATRIX, GROUND_DEPTH_TOLERANCE_MM, OBSTACLE_DETECTION_ENABLED
)
from utils import clamp

def get_warp_matrix():
    return np.array(WARP_MATRIX, dtype=np.float32)

def warp_frame(frame, M):
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
    debug_info["line_confidence"] = clamp(line_area / max(float(MIN_LINE_AREA * 8), 1.0), 0.0, 1.0)
    if line_area < MIN_LINE_AREA:
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
