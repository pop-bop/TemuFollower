import math
import cv2
import numpy as np


# ------------------------------------------------------------------
# VISION AGENT
# Classic contour-based line following with edge-width validation
# (10-12 px between line edges) and red-line stop detection.
# Integrates Dana's improved line classification for the mask.
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
        # ---- LINE TRACKING CONSTANTS ----
        self.LINE_WIDTH_MIN = 5
        self.LINE_WIDTH_MAX = 20

        # ---- ZERO SCAN (Baeyens derivative scanline) ----
        self.SCAN_HEIGHT_RATIO = 0.92
        self.SCAN_RADIUS = 140

        # ---- DANA LINE CLASSIFICATION ----
        self.WIDE_ROI_Y_START_RATIO = 0.20
        self.WIDE_ROI_X_START_RATIO = 0.0
        self.WIDE_ROI_X_END_RATIO = 1.0
        self.MIN_LINE_AREA = 40

        self.APPROACH_INTERSECTION_WIDTH_RATIO = 0.18
        self.INTERSECTION_WIDTH_RATIO = 0.30

        self.MARKER_GREEN_LOWER = (35, 80, 70)
        self.MARKER_GREEN_UPPER = (90, 255, 255)
        self.MARKER_RED_LOWER_1 = (0, 90, 80)
        self.MARKER_RED_UPPER_1 = (10, 255, 255)
        self.MARKER_RED_LOWER_2 = (165, 90, 80)
        self.MARKER_RED_UPPER_2 = (180, 255, 255)

        self.BLACK_THRESHOLD = 82
        self.STRICT_BLACK_THRESHOLD = 70

        # ---- RED LINE STOP ----
        self.RED_LINE_MIN_AREA = 60
        self.RED_LINE_STOP_FRAMES = 24
        self._red_stop_counter = 0

        # Frame-to-frame smoothing for seed
        self._seed_smoothing = 0.55
        self._prev_seed = None

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
    # Dana-style improved line mask
    # --------------------------------------------------------------
    def _improve_line_mask(self, gray_roi, color_roi=None):
        blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        _, mask_global = cv2.threshold(blurred, self.BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

        mask_adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 8
        )

        mask_color = np.zeros_like(gray_roi)
        mask_color[gray_roi < self.STRICT_BLACK_THRESHOLD] = 255

        mask_combined = cv2.bitwise_or(mask_global, mask_adaptive)
        mask_combined = cv2.bitwise_or(mask_combined, mask_color)

        if color_roi is not None:
            hsv = cv2.cvtColor(color_roi, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv, np.array(self.MARKER_GREEN_LOWER), np.array(self.MARKER_GREEN_UPPER))
            red_mask = cv2.bitwise_or(
                cv2.inRange(hsv, np.array(self.MARKER_RED_LOWER_1), np.array(self.MARKER_RED_UPPER_1)),
                cv2.inRange(hsv, np.array(self.MARKER_RED_LOWER_2), np.array(self.MARKER_RED_UPPER_2)),
            )
            mask_combined[green_mask > 0] = 0
            mask_combined[red_mask > 0] = 0

        kernel_open = np.ones((3, 3), np.uint8)
        mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel_open)
        kernel_close = np.ones((9, 9), np.uint8)
        mask_closed = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel_close)

        return mask_combined, mask_closed, mask_global, mask_adaptive, mask_color

    def _locate_line_simple(self, gray_roi, color_roi=None):
        mask_black, _, _, _, _ = self._improve_line_mask(gray_roi, color_roi)
        contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.MIN_LINE_AREA:
            return None
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return cx, cy

    # --------------------------------------------------------------
    # Zero-scan: Baeyens derivative scanline
    # --------------------------------------------------------------
    def _scanline_seed(self, gray, scan_y):
        h, w = gray.shape
        center_x = w // 2
        radius = min(self.SCAN_RADIUS, center_x, w - 1 - center_x)
        if radius < 2:
            return None
        scan_data = gray[scan_y, center_x - radius: center_x + radius].astype(np.float32)

        der = np.zeros_like(scan_data)
        der[1:-1] = scan_data[:-2] - scan_data[2:]

        left_edge = int(np.argmax(der))
        right_edge = int(np.argmin(der))
        line_idx = (left_edge + right_edge) / 2.0

        line_x = int(round(center_x - radius + line_idx))
        line_x = max(0, min(w - 1, line_x))
        return line_x, scan_y

    # --------------------------------------------------------------
    # Find line contour with width validation
    # --------------------------------------------------------------
    def _find_line_contour(self, mask_black):
        """Return the largest contour above MIN_LINE_AREA (Dana style).
        No width/geometry filtering — a curved line at a turn naturally
        produces a valid contour whose area-weighted centroid is exactly
        what the PID needs to steer toward it. Width ratio is checked
        later in process_frame for intersection detection."""
        contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best = None
        best_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.MIN_LINE_AREA:
                continue
            if area > best_area:
                best_area = area
                best = c

        return best

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
        found = self._locate_line_simple(gray_wide, wide_roi)
        if found is None:
            return False

        cx, cy = found
        real_x = cx + wx_start
        real_y = cy + wy_start
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
        # Blue centre reference line (Dana style)
        roi_center_x = roi_x_start + roi_w // 2
        cv2.line(frame, (roi_center_x, roi_start_y), (roi_center_x, h), (255, 0, 0), 1)
        cv2.putText(frame, "ROI", (roi_x_start + 5, roi_start_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # ---- Step 1: Black line mask (Dana) ----
        mask_black, mask_closed, mask_global, mask_adaptive, mask_color = \
            self._improve_line_mask(gray_roi, roi)

        # Debug layers
        b_ch, g_ch, r_ch = cv2.split(roi)
        color_threshold = 50
        no_color = np.zeros_like(b_ch)
        no_color[(b_ch < color_threshold) & (g_ch < color_threshold) & (r_ch < color_threshold)] = 255
        result["cnn_layers"] = {
            "Black_Mask": mask_black,
            "Mask_Closed": mask_closed,
            "R_Channel": r_ch,
            "G_Channel": g_ch,
            "B_Channel": b_ch,
            "No_Color_Channel": no_color,
        }

        # ---- Step 2: Find line via contour analysis ----
        line_contour = self._find_line_contour(mask_black)

        # ---- Step 3: Red line detection (bottom-centre of ROI) ----
        red_line_now = self._detect_red_at_bottom(roi)
        if red_line_now:
            self._red_stop_counter = self.RED_LINE_STOP_FRAMES
        if self._red_stop_counter > 0:
            result["red_line_detected"] = True
            result["special_state"] = "red_line"
            self._red_stop_counter -= 1
            cv2.putText(frame, "RED LINE STOP", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if line_contour is not None and not result.get("red_line_detected"):
            M = cv2.moments(line_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]) + roi_x_start
                cy = int(M["m01"] / M["m00"])
                result["line_center_x"] = cx
                result["line_center_y"] = cy + roi_start_y

                # Centroid marker
                cv2.circle(frame, (cx, cy + roi_start_y), 4, (0, 0, 255), -1)

                # Heading arrow from fitLine
                shifted = line_contour + np.array([roi_x_start, roi_start_y])
                if len(shifted) >= 5:
                    ellipse = cv2.fitEllipse(shifted)
                    cv2.ellipse(frame, ellipse, (255, 100, 0), 2)
                    [vx, vy, _, _] = cv2.fitLine(shifted, cv2.DIST_L2, 0, 0.01, 0.01)
                    vx = vx.item()
                    vy = vy.item()
                    if vy > 0:
                        vx = -vx
                        vy = -vy
                    length = 50
                    pt2 = (int(cx + vx * length), int(cy + roi_start_y + vy * length))
                    cv2.arrowedLine(frame, (cx, cy + roi_start_y), pt2,
                                    (0, 255, 255), 3, tipLength=0.3)
                else:
                    cv2.drawContours(frame, [shifted], -1, (255, 100, 0), 2)

                # Intersection detection via bounding box width ratio
                x_box, y_box, w_line, h_line = cv2.boundingRect(line_contour)
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
            # ---- Step 4: No line found — try seed / fallback ----
            scan_y = min(roi_h - 2, int(roi_h * self.SCAN_HEIGHT_RATIO))
            seed = self._scanline_seed(gray_roi, scan_y)
            if seed is not None:
                if self._prev_seed is not None:
                    sx = int(round(self._seed_smoothing * seed[0] + (1.0 - self._seed_smoothing) * self._prev_seed[0]))
                    seed = (sx, seed[1])
                self._prev_seed = (seed[0], seed[1])
                result["line_center_x"] = seed[0] + roi_x_start
                result["line_center_y"] = seed[1] + roi_start_y
                cv2.circle(frame, (seed[0] + roi_x_start, seed[1] + roi_start_y), 5, (0, 255, 0), -1)
                cv2.putText(frame, f"Seed:{seed[0]}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            elif not self._try_wide_roi_fallback(frame, result):
                cls = self._classify_line_end(mask_black, roi_h, roi_w, frame)
                result.update(cls)

        return result
