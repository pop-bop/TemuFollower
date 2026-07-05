import math

import cv2

from config import ROI_VIEW_SCALE, ARROW_MAX_DEFLECTION_DEG, MAX_SPEED
from utils import clamp


def draw_debug_view(frame, debug_info, error, turn, left, right, state,
                     extra_lines=None, lookahead_bounds=None, lookahead_point=None):
    x0, y0, x1, y1 = debug_info["roi_bounds"]
    debug = frame.copy()

    cv2.rectangle(debug, (x0, y0), (x1, y1), (0, 255, 255), 1)
    center_x = x0 + (x1 - x0) // 2
    cv2.line(debug, (center_x, y0), (center_x, y1), (255, 0, 0), 1)

    if lookahead_bounds is not None:
        lx0, ly0, lx1, ly1 = lookahead_bounds
        cv2.rectangle(debug, (lx0, ly0), (lx1, ly1), (255, 128, 0), 1)

    if debug_info["line_point"] is not None:
        cv2.circle(debug, debug_info["line_point"], 5, (0, 0, 255), -1)

    if lookahead_point is not None:
        cv2.circle(debug, lookahead_point, 5, (255, 128, 0), -1)

    lines = [
        f"state: {state}",
        f"error: {error:+.3f}" if error is not None else "error: none",
        f"turn:  {turn:+.3f}" if turn is not None else "turn:  none",
        f"L/R:   {left:+.2f} / {right:+.2f}" if left is not None else "L/R:   none",
    ]
    if extra_lines:
        lines.extend(extra_lines)
    for i, text in enumerate(lines):
        cv2.putText(debug, text, (8, 18 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.imshow("Camera + Decisions", debug)
    cv2.imshow("Black Mask", debug_info["mask"])


def draw_roi_arrow_view(debug_info, applied_left, applied_right, state, lean_ratio=None):
    roi = debug_info["roi_frame"]
    if roi is None or roi.size == 0:
        return

    view = cv2.resize(
        roi,
        (roi.shape[1] * ROI_VIEW_SCALE, roi.shape[0] * ROI_VIEW_SCALE),
        interpolation=cv2.INTER_NEAREST,
    )
    h, w = view.shape[:2]

    if lean_ratio is None:
        diff = applied_left - applied_right
        span = max(0.001, 2.0 * MAX_SPEED)
        lean_ratio = diff / span

    lean_ratio = clamp(lean_ratio, -1.0, 1.0)
    angle_deg = lean_ratio * ARROW_MAX_DEFLECTION_DEG
    angle_rad = math.radians(angle_deg)

    arrow_length = min(h, w) * 0.42
    base_x, base_y = w // 2, h - 10
    tip_x = int(base_x + arrow_length * math.sin(angle_rad))
    tip_y = int(base_y - arrow_length * math.cos(angle_rad))

    cv2.arrowedLine(view, (base_x, base_y), (tip_x, tip_y), (0, 255, 0), 3, tipLength=0.3)

    cv2.putText(view, f"L={applied_left:+.2f}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(view, f"R={applied_right:+.2f}", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(view, f"state={state}", (8, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    cv2.imshow("ROI + Direction Arrow", view)
