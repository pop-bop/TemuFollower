"""Bird's-eye rectification of the ground plane.

Projects a known metric ground patch to a fixed-scale top-down image. Two things
fall out of that:

  * ROI rows become true distances in mm instead of image ratios, so the near /
    lookahead / far ROIs stop meaning different things at different speeds.
  * Optical flow in the rectified image is metrically scaled by construction
    (BEV_MM_PER_PX), which is how a monocular camera recovers absolute speed
    without wheel encoders. Ordinary monocular VO cannot do this -- the scale
    comes from knowing the ground plane.
"""

import json
import math
import os

import cv2
import numpy as np

from config import (
    CAMERA_TILT_ANGLE_DEG, CAMERA_MOUNT_HEIGHT_MM,
    BEV_NEAR_MM, BEV_FAR_MM, BEV_WIDTH_MM, BEV_OUTPUT_PX,
    BEV_CAMERA_X_OFFSET_MM, BEV_CAMERA_AXLE_OFFSET_MM,
    BEV_CALIBRATION_FILE,
)


def ground_to_pixel(x_mm, y_mm, intrinsics):
    """Ground point -> image pixel.

    Ground frame: +y forward from the front axle, +x right, z=0 at the floor.
    Camera frame: +X right, +Y down, +Z forward along the optical axis.
    """
    tilt = math.radians(CAMERA_TILT_ANGLE_DEG)

    # Ground point relative to the lens.
    fwd = y_mm + BEV_CAMERA_AXLE_OFFSET_MM
    right = x_mm - BEV_CAMERA_X_OFFSET_MM
    down = CAMERA_MOUNT_HEIGHT_MM

    # Rotate by the tilt: the camera is pitched down, so rotate about its X axis.
    cam_z = fwd * math.cos(tilt) + down * math.sin(tilt)
    cam_y = -fwd * math.sin(tilt) + down * math.cos(tilt)
    cam_x = right

    if cam_z <= 1e-6:
        return None

    u = intrinsics.ppx + (cam_x / cam_z) * intrinsics.fx
    v = intrinsics.ppy + (cam_y / cam_z) * intrinsics.fy
    return (u, v)


def _output_size():
    height_mm = BEV_FAR_MM - BEV_NEAR_MM
    h_px = int(round(height_mm / (BEV_WIDTH_MM / BEV_OUTPUT_PX)))
    return BEV_OUTPUT_PX, h_px


def compute_bev_matrix(intrinsics):
    """Analytic homography from mount geometry + live intrinsics.

    Returns (M, (w, h)) or (None, None) if the patch is not fully imageable.
    """
    w_px, h_px = _output_size()
    half_w = BEV_WIDTH_MM / 2.0

    # Ground corners: near-left, near-right, far-right, far-left.
    corners_mm = [
        (-half_w, BEV_NEAR_MM),
        (half_w, BEV_NEAR_MM),
        (half_w, BEV_FAR_MM),
        (-half_w, BEV_FAR_MM),
    ]
    # In the output, +y forward maps to decreasing row (far at the top).
    corners_px = [
        (0.0, float(h_px)),
        (float(w_px), float(h_px)),
        (float(w_px), 0.0),
        (0.0, 0.0),
    ]

    src = []
    for x_mm, y_mm in corners_mm:
        p = ground_to_pixel(x_mm, y_mm, intrinsics)
        if p is None:
            return None, None
        src.append(p)

    M = cv2.getPerspectiveTransform(
        np.array(src, dtype=np.float32),
        np.array(corners_px, dtype=np.float32),
    )
    return M, (w_px, h_px)


def load_bev_matrix(intrinsics, path=BEV_CALIBRATION_FILE):
    """Measured calibration if present, else the analytic derivation."""
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            M = np.array(data["matrix"], dtype=np.float32)
            size = tuple(data["output_size"])
            print(f"BEV: using measured calibration from {path}")
            return M, size
        except (ValueError, KeyError) as e:
            print(f"BEV: {path} unreadable ({e}), falling back to analytic")

    if intrinsics is None:
        print("BEV: no intrinsics available, rectification disabled")
        return None, None

    M, size = compute_bev_matrix(intrinsics)
    if M is None:
        print("BEV: patch not fully imageable at this mount geometry")
        return None, None
    print(f"BEV: analytic homography, {size[0]}x{size[1]}px "
          f"for {BEV_WIDTH_MM:.0f}x{BEV_FAR_MM - BEV_NEAR_MM:.0f}mm patch")
    return M, size


def warp_to_bev(frame, M, size, interpolation=cv2.INTER_LINEAR):
    return cv2.warpPerspective(frame, M, size, flags=interpolation)


def bev_validity_mask(M, size, frame_shape):
    """Which BEV pixels come from real image data.

    The near corners of the patch fall outside the camera's horizontal FOV, so
    they warp in as black. Treating that black as line would drag steering.
    """
    h, w = frame_shape[:2]
    src = np.zeros((h, w), dtype=np.uint8)
    src[:] = 255
    mask = cv2.warpPerspective(src, M, size, flags=cv2.INTER_NEAREST)
    return mask


def row_to_distance_mm(row, size):
    """BEV row index -> mm ahead of the front axle."""
    _, h_px = size
    frac = 1.0 - (row / max(1.0, float(h_px)))
    return BEV_NEAR_MM + frac * (BEV_FAR_MM - BEV_NEAR_MM)


def distance_to_row(distance_mm, size):
    _, h_px = size
    frac = (distance_mm - BEV_NEAR_MM) / max(1e-6, BEV_FAR_MM - BEV_NEAR_MM)
    return int(round((1.0 - frac) * h_px))
