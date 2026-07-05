#!/usr/bin/env python3
"""
Simple line detection test for the DPSI-LFR robot.

Reads camera frames, finds the black line in a bottom region of
interest, and prints its position. No motors, no PID, no markers,
just camera in and line position out.

Run with:
    python3 line_detect_test.py

Press q in the debug window to quit.
"""

import cv2
import numpy as np

CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 120

ROI_Y_START_RATIO = 0.48
ROI_X_START_RATIO = 0.05
ROI_X_END_RATIO = 0.95

BLACK_THRESHOLD = 82
MIN_LINE_AREA = 45


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


def find_line(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, black_mask = cv2.threshold(blur, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, black_mask

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_LINE_AREA:
        return None, black_mask

    M = cv2.moments(largest)
    if M["m00"] <= 0:
        return None, black_mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), black_mask


def main():
    camera_kind, camera = open_camera()

    print("Line detect test running. Press q in the debug window to quit.")

    try:
        while True:
            frame = read_frame(camera_kind, camera)
            if frame is None:
                print("No frame received")
                continue

            h, w = frame.shape[:2]
            y0 = int(h * ROI_Y_START_RATIO)
            x0 = int(w * ROI_X_START_RATIO)
            x1 = int(w * ROI_X_END_RATIO)
            roi = frame[y0:h, x0:x1]
            roi_w = x1 - x0
            center_x_global = x0 + roi_w // 2

            line_point, black_mask = find_line(roi)

            debug = frame.copy()
            cv2.rectangle(debug, (x0, y0), (x1, h), (0, 255, 255), 1)
            cv2.line(debug, (center_x_global, y0), (center_x_global, h), (255, 0, 0), 1)

            if line_point is not None:
                cx_roi, cy_roi = line_point
                cx_global = cx_roi + x0
                cy_global = cy_roi + y0
                error = (cx_global - center_x_global) / max(1.0, roi_w / 2.0)
                print(f"line found  x={cx_global}  error={error:+.3f}")
                cv2.circle(debug, (cx_global, cy_global), 5, (0, 0, 255), -1)
            else:
                print("line not found")

            cv2.imshow("Line Detect Debug", debug)
            cv2.imshow("Black Mask", black_mask)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        if camera_kind == "picamera2":
            camera.stop()
        else:
            camera.release()
        cv2.destroyAllWindows()
        print("Camera closed")


if __name__ == "__main__":
    main()