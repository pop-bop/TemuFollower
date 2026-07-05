import cv2
import numpy as np


# ------------------------------------------------------------------
# VISION AGENT
# Dana's line detection (simple threshold + morph + largest contour),
# with intersection detection and red-line stop layered on top.
# ------------------------------------------------------------------
class VisionAgent:
    def __init__(self, resolution=(320, 240), fps=30, open_camera=True):
        self.resolution = resolution
        self.fps = fps
        self.camera_kind = None
        self.camera = None
        self.simulation_mode = False

        if not open_camera:
            self.simulation_mode = True
            self._init_constants()
            return

        try:
            from picamera2 import Picamera2
            picam2 = Picamera2()
            config = picam2.create_video_configuration(
                main={"size": resolution, "format": "RGB888"},
                controls={"FrameRate": fps},
            )
            picam2.configure(config)
            picam2.start()
            self.camera_kind = "picamera2"
            self.camera = picam2
            print(f"[Vision] Using Picamera2 at {resolution[0]}x{resolution[1]} @ {fps}fps")
        except Exception as exc:
            print(f"[Vision] Picamera2 unavailable: {exc}")
            print("[Vision] Falling back to cv2.VideoCapture...")
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            cap.set(cv2.CAP_PROP_FPS, fps)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    self.camera_kind = "cv2"
                    self.camera = cap
                    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    actual_fps = cap.get(cv2.CAP_PROP_FPS)
                    print(f"[Vision] Camera opened via cv2 ({actual_w}x{actual_h} @ {actual_fps}fps)")
                else:
                    cap.release()
            else:
                cap.release()

        if self.camera is None:
            print("[Vision] No camera found. Running in simulation mode.")
            self.simulation_mode = True

        self._init_constants()

    def _init_constants(self):
        # ---- DANA LINE TRACKING CONSTANTS ----
        self.BLACK_THRESHOLD = 82
        self.MIN_LINE_AREA = 45

        self.MARKER_GREEN_LOWER = (35, 80, 70)
        self.MARKER_GREEN_UPPER = (90, 255, 255)
        self.MARKER_RED_LOWER_1 = (0, 90, 80)
        self.MARKER_RED_UPPER_1 = (10, 255, 255)
        self.MARKER_RED_LOWER_2 = (165, 90, 80)
        self.MARKER_RED_UPPER_2 = (180, 255, 255)

        # ---- WIDE ROI (approach fallback) ----
        self.WIDE_ROI_Y_START_RATIO = 0.20
        self.WIDE_ROI_X_START_RATIO = 0.0
        self.WIDE_ROI_X_END_RATIO = 1.0

        # ---- INTERSECTION DETECTION ----
        self.APPROACH_INTERSECTION_WIDTH_RATIO = 0.18
        self.INTERSECTION_WIDTH_RATIO = 0.30

        # ---- RED LINE STOP ----
        self.RED_LINE_MIN_AREA = 60
        self.RED_LINE_STOP_FRAMES = 24
        self._red_stop_counter = 0

    # --------------------------------------------------------------
    # Camera I/O
    # --------------------------------------------------------------
    def get_frame(self):
        if self.simulation_mode:
            print("[Vision] Simulation mode — generating test frames. Press 'q' in OpenCV window to exit.")
            while True:
                test = np.ones((self.resolution[1], self.resolution[0], 3), dtype=np.uint8) * 220
                cx = self.resolution[0] // 2
                cv2.line(test, (cx, int(self.resolution[1] * 0.4)),
                         (cx + 15, self.resolution[1]), (20, 20, 20), 4)
                noise = np.random.randint(0, 15, test.shape, dtype=np.uint8)
                test = cv2.add(test, noise)
                yield test
        else:
            while True:
                frame = self._read_frame()
                if frame is None:
                    print("[Vision] Camera read failed. Reconnecting...")
                    self._reconnect_camera()
                    if self.camera is None:
                        break
                    continue
                yield frame

    def _read_frame(self):
        if self.camera_kind == "picamera2":
            try:
                rgb = self.camera.capture_array()
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            except Exception:
                return None
        elif self.camera_kind == "cv2":
            ret, frame = self.camera.read()
            return frame if ret else None
        return None

    def _reconnect_camera(self):
        if self.camera is not None:
            try:
                if self.camera_kind == "picamera2":
                    self.camera.stop()
                else:
                    self.camera.release()
            except Exception:
                pass
        import time
        time.sleep(1.0)

        try:
            from picamera2 import Picamera2
            picam2 = Picamera2()
            config = picam2.create_video_configuration(
                main={"size": self.resolution, "format": "RGB888"},
                controls={"FrameRate": self.fps},
            )
            picam2.configure(config)
            picam2.start()
            self.camera_kind = "picamera2"
            self.camera = picam2
            print(f"[Vision] Reconnected via Picamera2 at {self.resolution[0]}x{self.resolution[1]} @ {self.fps}fps")
            return
        except Exception as exc:
            print(f"[Vision] Picamera2 reconnect failed: {exc}")

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                self.camera_kind = "cv2"
                self.camera = cap
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"[Vision] Reconnected via cv2 ({actual_w}x{actual_h} @ {actual_fps}fps)")
                return
            cap.release()

        print("[Vision] Reconnect failed. No camera available.")
        self.camera = None
        self.camera_kind = None

    # --------------------------------------------------------------
    # Dana's line detection: blur → threshold → exclude markers → morph
    # --------------------------------------------------------------
    def _line_mask(self, gray_roi, color_roi=None):
        """Dana's mask generation — single global threshold (no hybrid)."""
        blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        _, mask = cv2.threshold(blurred, self.BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

        if color_roi is not None:
            hsv = cv2.cvtColor(color_roi, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv, np.array(self.MARKER_GREEN_LOWER), np.array(self.MARKER_GREEN_UPPER))
            red_mask = cv2.bitwise_or(
                cv2.inRange(hsv, np.array(self.MARKER_RED_LOWER_1), np.array(self.MARKER_RED_UPPER_1)),
                cv2.inRange(hsv, np.array(self.MARKER_RED_LOWER_2), np.array(self.MARKER_RED_UPPER_2)),
            )
            mask[green_mask > 0] = 0
            mask[red_mask > 0] = 0

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        return mask

    def _dana_find_line(self, gray_roi, color_roi=None):
        """Dana's line detection: mask → largest contour → centroid.
        Returns ((cx_roi, cy_roi), mask) or (None, mask)."""
        mask = self._line_mask(gray_roi, color_roi)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, mask

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.MIN_LINE_AREA:
            return None, mask

        M = cv2.moments(largest)
        if M["m00"] <= 0:
            return None, mask

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy, largest), mask

    # --------------------------------------------------------------
    # Red line detection (bottom of ROI)
    # --------------------------------------------------------------
    def _detect_red_at_bottom(self, roi):
        """Check whether a red line/region is present in the bottom
        centre of the ROI (where the robot's front wheels are)."""
        h, w = roi.shape[:2]
        bottom_strip = roi[int(h * 0.7):, int(w * 0.2):int(w * 0.8)]
        if bottom_strip.size == 0:
            return False
        hsv = cv2.cvtColor(bottom_strip, cv2.COLOR_BGR2HSV)
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array(self.MARKER_RED_LOWER_1), np.array(self.MARKER_RED_UPPER_1)),
            cv2.inRange(hsv, np.array(self.MARKER_RED_LOWER_2), np.array(self.MARKER_RED_UPPER_2)),
        )
        red_area = cv2.countNonZero(red_mask)
        return red_area > self.RED_LINE_MIN_AREA

    # --------------------------------------------------------------
    # Dashed / dead-end classifier (Dana style)
    # --------------------------------------------------------------
    def _classify_line_end(self, mask_black, roi_h, roi_w, frame):
        result = {}

        scan_cols = [roi_w // 4, roi_w // 2, 3 * roi_w // 4]
        total_top = 0
        total_bottom = 0

        for sx in scan_cols:
            strip = mask_black[:, max(0, sx - 3):min(roi_w, sx + 3)]
            col_sum = np.sum(strip > 0, axis=1)
            mid = roi_h // 2
            total_top += np.sum(col_sum[:mid] > 0)
            total_bottom += np.sum(col_sum[mid:] > 0)

        total_pixels = np.sum(mask_black > 0)
        black_ratio = total_pixels / (roi_h * roi_w) if roi_h * roi_w > 0 else 0

        row_coverage = np.sum(mask_black > 0, axis=1)
        nonzero_rows = np.where(row_coverage > 0)[0]
        if len(nonzero_rows) > 0:
            span_ratio = (nonzero_rows[-1] - nonzero_rows[0]) / max(1, roi_h)
        else:
            span_ratio = 0

        if total_top > 0 and black_ratio > 0.015:
            result["is_dashed"] = True
            result["special_state"] = "dashed_gap"
            cv2.putText(frame, "DASHED GAP", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        elif total_bottom > 0 and total_top == 0 and span_ratio < 0.5:
            result["line_ended"] = True
            result["special_state"] = "dead_end"
            cv2.putText(frame, "DEAD END (180)", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif total_bottom > 0 and span_ratio >= 0.5:
            result["line_ended"] = True
            result["special_state"] = "dead_end"
            cv2.putText(frame, "DEAD END (180)", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            if black_ratio > 0.003:
                result["is_dashed"] = True
                result["special_state"] = "dashed_gap"
                cv2.putText(frame, "DASHED GAP", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                result["line_ended"] = True
                result["special_state"] = "dead_end"
                cv2.putText(frame, "DEAD END (180)", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return result

    # --------------------------------------------------------------
    # Wide ROI fallback (approach mode)
    # --------------------------------------------------------------
    def _try_wide_roi_fallback(self, frame, result):
        h, w = frame.shape[:2]
        wy_start = int(h * self.WIDE_ROI_Y_START_RATIO)
        wx_start = int(w * self.WIDE_ROI_X_START_RATIO)
        wx_end = int(w * self.WIDE_ROI_X_END_RATIO)

        wide_roi = frame[wy_start:, wx_start:wx_end]
        if wide_roi.size == 0:
            return False

        gray_wide = cv2.cvtColor(wide_roi, cv2.COLOR_BGR2GRAY)
        found, _ = self._dana_find_line(gray_wide, wide_roi)
        if found is None:
            return False

        cx_roi, cy_roi, _ = found
        real_x = cx_roi + wx_start
        real_y = cy_roi + wy_start
        result["line_center_x"] = real_x
        result["line_center_y"] = real_y
        result["special_state"] = "approach"
        cv2.circle(frame, (real_x, real_y), 6, (255, 0, 255), -1)
        cv2.putText(frame, "APPROACH", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        return True

    # --------------------------------------------------------------
    # MAIN FRAME PROCESSING
    # --------------------------------------------------------------
    def process_frame(self, frame):
        result = {
            "line_center_x": None,
            "line_center_y": None,
            "line_ended": False,
            "is_dashed": False,
            "special_state": None,
            "line_curvature": 0.0,
            "cnn_layers": None,
            "red_line_detected": False,
        }

        h, w = frame.shape[:2]

        # ROI: bottom portion, cropped 20 px from each side
        roi_start_y = int(h * 0.55)
        roi_x_start = 20
        roi_x_end = max(roi_x_start + 1, w - 20)
        roi = frame[roi_start_y:, roi_x_start:roi_x_end]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_h, roi_w = gray_roi.shape

        cv2.rectangle(frame, (roi_x_start, roi_start_y), (roi_x_end, h), (0, 255, 255), 2)
        roi_center_x = roi_x_start + roi_w // 2
        cv2.line(frame, (roi_center_x, roi_start_y), (roi_center_x, h), (255, 0, 0), 1)
        cv2.putText(frame, "ROI", (roi_x_start + 5, roi_start_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # ---- Step 1: Dana line detection ----
        found, mask = self._dana_find_line(gray_roi, roi)

        # Debug layer (single mask)
        result["cnn_layers"] = {"Mask": mask}

        # ---- Step 2: Red line detection (bottom-centre of ROI) ----
        red_line_now = self._detect_red_at_bottom(roi)
        if red_line_now:
            self._red_stop_counter = self.RED_LINE_STOP_FRAMES
        if self._red_stop_counter > 0:
            result["red_line_detected"] = True
            result["special_state"] = "red_line"
            self._red_stop_counter -= 1
            cv2.putText(frame, "RED LINE STOP", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # ---- Step 3: Process line data ----
        if found is not None and not result.get("red_line_detected"):
            cx_roi, cy_roi, contour = found
            cx = cx_roi + roi_x_start
            cy = cy_roi
            result["line_center_x"] = cx
            result["line_center_y"] = cy + roi_start_y

            cv2.circle(frame, (cx, cy + roi_start_y), 4, (0, 0, 255), -1)

            # Intersection detection via bounding rect width
            x_box, y_box, w_line, h_line = cv2.boundingRect(contour)
            width_ratio = w_line / max(1, roi_w)
            if width_ratio > self.INTERSECTION_WIDTH_RATIO:
                result["special_state"] = "intersection"
                cv2.putText(frame, "INTERSECTION", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            elif width_ratio > self.APPROACH_INTERSECTION_WIDTH_RATIO:
                result["special_state"] = "approaching_intersection"
                cv2.putText(frame, "APPROACHING INTERSECTION", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            cv2.putText(frame, f"CX:{cx} W:{w_line}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        elif not result.get("red_line_detected"):
            # ---- Step 4: No line — wide ROI fallback, then classify ----
            if not self._try_wide_roi_fallback(frame, result):
                cls = self._classify_line_end(mask, roi_h, roi_w, frame)
                result.update(cls)

        return result
