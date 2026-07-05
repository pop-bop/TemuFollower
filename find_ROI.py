#!/usr/bin/env python3
"""
ROI finder for the DPSI-LFR robot.

Shows a live camera preview with an adjustable ROI box. Use the
keyboard to move each edge of the box, then copy the printed ratio
values into line_follow_pid.py.

Run with:
    python3 roi_finder.py

Controls:
    w / s  = move top edge up / down
    a / d  = move left edge left / right
    j / l  = move right edge left / right
    i / k  = move bottom edge up / down
    p      = print current ROI ratio values
    q      = quit
"""

import cv2

CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 120

BLACK_THRESHOLD = 82
STEP = 0.01

# Starting values, same as line_follow_pid.py defaults
roi_y_start_ratio = 0.48
roi_y_end_ratio = 1.00
roi_x_start_ratio = 0.05
roi_x_end_ratio = 0.95


def open_camera():
    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        config = picam2.create_video_configuration(
            main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"},
            controls={"FrameRate": CAMERA_FPS},
        )
        picam2.configure(config)
        picam2.start()
        print("Using Picamera2")
        return "picamera2", picam2
    except Exception as exc:
        print(f"Picamera2 unavailable, falling back to cv2.VideoCapture: {exc}")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        print("Using cv2.VideoCapture")
        return "cv2", cap


def read_frame(camera_kind, camera):
    if camera_kind == "picamera2":
        rgb = camera.capture_array()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        ok, frame = camera.read()
        return frame if ok else None


def clamp01(value):
    return max(0.0, min(1.0, value))


def print_values():
    print("Current ROI values:")
    print(f"  ROI_Y_START_RATIO = {roi_y_start_ratio:.2f}")
    print(f"  ROI_Y_END_RATIO   = {roi_y_end_ratio:.2f}")
    print(f"  ROI_X_START_RATIO = {roi_x_start_ratio:.2f}")
    print(f"  ROI_X_END_RATIO   = {roi_x_end_ratio:.2f}")


def main():
    global roi_y_start_ratio, roi_y_end_ratio, roi_x_start_ratio, roi_x_end_ratio

    camera_kind, camera = open_camera()

    print("ROI finder running.")
    print("w/s = top edge, a/d = left edge, j/l = right edge, i/k = bottom edge")
    print("p = print values, q = quit")

    try:
        while True:
            frame = read_frame(camera_kind, camera)
            if frame is None:
                continue

            h, w = frame.shape[:2]
            y0 = int(h * roi_y_start_ratio)
            y1 = int(h * roi_y_end_ratio)
            x0 = int(w * roi_x_start_ratio)
            x1 = int(w * roi_x_end_ratio)

            debug = frame.copy()
            cv2.rectangle(debug, (x0, y0), (x1, y1), (0, 255, 255), 2)
            center_x = x0 + (x1 - x0) // 2
            cv2.line(debug, (center_x, y0), (center_x, y1), (255, 0, 0), 1)
            cv2.putText(
                debug,
                f"y:{roi_y_start_ratio:.2f}-{roi_y_end_ratio:.2f} x:{roi_x_start_ratio:.2f}-{roi_x_end_ratio:.2f}",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )
            cv2.imshow("ROI Finder", debug)

            if y1 > y0 and x1 > x0:
                roi = frame[y0:y1, x0:x1]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                _, black_mask = cv2.threshold(blur, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
                cv2.imshow("ROI Black Mask", black_mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("w"):
                roi_y_start_ratio = clamp01(roi_y_start_ratio - STEP)
            elif key == ord("s"):
                roi_y_start_ratio = clamp01(roi_y_start_ratio + STEP)
            elif key == ord("i"):
                roi_y_end_ratio = clamp01(roi_y_end_ratio - STEP)
            elif key == ord("k"):
                roi_y_end_ratio = clamp01(roi_y_end_ratio + STEP)
            elif key == ord("a"):
                roi_x_start_ratio = clamp01(roi_x_start_ratio - STEP)
            elif key == ord("d"):
                roi_x_start_ratio = clamp01(roi_x_start_ratio + STEP)
            elif key == ord("j"):
                roi_x_end_ratio = clamp01(roi_x_end_ratio - STEP)
            elif key == ord("l"):
                roi_x_end_ratio = clamp01(roi_x_end_ratio + STEP)
            elif key == ord("p"):
                print_values()

    finally:
        if camera_kind == "picamera2":
            camera.stop()
        else:
            camera.release()
        cv2.destroyAllWindows()
        print("Camera closed")
        print_values()


if __name__ == "__main__":
    main()
