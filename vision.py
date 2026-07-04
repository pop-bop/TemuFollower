import cv2
import numpy as np

class VisionAgent:
    def __init__(self, resolution=(320, 240), fps=30):
        self.resolution = resolution
        self.fps = fps
        self.camera_kind = None
        self.camera = None
        self.simulation_mode = False

        # Try Picamera2 first (works on Raspberry Pi), then fall back to cv2.VideoCapture
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

        # Wide/far search ROI: checked when the close-in ROI loses the line,
        # so a curve or gap doesn't immediately look like a dead end.
        self.WIDE_ROI_Y_START_RATIO = 0.15
        self.WIDE_ROI_X_START_RATIO = 0.0
        self.WIDE_ROI_X_END_RATIO = 1.0
        self.MIN_LINE_AREA = 40

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
        """Read a frame from the camera, handling both picamera2 and cv2 backends."""
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
        """Try to reconnect to the camera after a failure."""
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

        # Try Picamera2 first
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

        # Fall back to cv2
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

    def _improve_line_mask(self, gray_roi):
        """
        Multi-method black line mask.
        Uses both global + adaptive thresholding for robustness.
        """
        # --- Method 1: Global threshold (works well for high-contrast lines) ---
        blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        _, mask_global = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY_INV)

        # --- Method 2: Adaptive threshold (handles shadows, uneven lighting) ---
        mask_adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 8
        )

        # --- Method 3: Strict intensity threshold (catches very dark pixels) ---
        mask_color = np.zeros_like(gray_roi)
        mask_color[gray_roi < 70] = 255

        # Combine: a pixel is "line" if ANY method says so (union)
        mask_combined = cv2.bitwise_or(mask_global, mask_adaptive)
        mask_combined = cv2.bitwise_or(mask_combined, mask_color)

        # Morphological cleanup
        kernel_open = np.ones((3, 3), np.uint8)
        mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel_open)

        # CLOSE to bridge small gaps in solid lines
        kernel_close = np.ones((5, 5), np.uint8)
        mask_closed = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel_close)

        return mask_combined, mask_closed, mask_global, mask_adaptive, mask_color

    def _locate_line_simple(self, gray_roi):
        """
        Simple largest-contour line finder used for the wide/far search ROI.
        Returns (cx, cy) in gray_roi-local coordinates, or None if nothing found.
        """
        mask_black, _, _, _, _ = self._improve_line_mask(gray_roi)
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

    def _try_wide_roi_fallback(self, frame, result):
        """
        When the close-in ROI loses the line, search a wider/farther-out ROI
        before concluding the line has actually ended. On success, marks
        special_state="approach" so the control layer can steer gently back
        toward the line instead of treating it as a dead end.
        """
        h, w = frame.shape[:2]
        wy_start = int(h * self.WIDE_ROI_Y_START_RATIO)
        wx_start = int(w * self.WIDE_ROI_X_START_RATIO)
        wx_end = int(w * self.WIDE_ROI_X_END_RATIO)

        wide_roi = frame[wy_start:, wx_start:wx_end]
        if wide_roi.size == 0:
            return False

        gray_wide = cv2.cvtColor(wide_roi, cv2.COLOR_BGR2GRAY)
        found = self._locate_line_simple(gray_wide)
        if found is None:
            return False

        cx, cy = found
        real_x = cx + wx_start
        real_y = cy + wy_start
        result["line_center_x"] = real_x
        result["line_center_y"] = real_y
        result["special_state"] = "approach"
        cv2.circle(frame, (real_x, real_y), 6, (255, 0, 255), -1)
        cv2.putText(frame, "APPROACH", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        return True

    def process_frame(self, frame):
        """
        Processes a single frame for line following and maze solving.
        Returns a dict containing line position and special states.
        """
        result = {
            "line_center_x": None,
            "line_center_y": None,
            "line_ended": False,
            "is_dashed": False,
            "special_state": None,
            "cnn_layers": None
        }

        h, w = frame.shape[:2]

        # ROI: bottom-middle 40% width, bottom 40% height
        roi_start_y = int(h * 0.6)
        roi_start_x = int(w * 0.3)
        roi_end_x = int(w * 0.7)
        roi = frame[roi_start_y:, roi_start_x:roi_end_x]
        roi_h = roi.shape[0]
        roi_w = roi.shape[1]

        cv2.rectangle(frame, (roi_start_x, roi_start_y), (roi_end_x, h), (0, 255, 255), 2)
        cv2.putText(frame, "ROI", (roi_start_x, roi_start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # --- IMPROVED BLACK LINE DETECTION ---
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        mask_black, mask_closed, mask_global, mask_adaptive, mask_color = \
            self._improve_line_mask(gray_roi)

        b, g, r = cv2.split(roi)
        color_threshold = 50
        no_color = np.zeros_like(b)
        no_color[(b < color_threshold) & (g < color_threshold) & (r < color_threshold)] = 255

        result["cnn_layers"] = {
            "Black_Mask": mask_black,
            "Mask_Closed": mask_closed,
            "R_Channel": r,
            "G_Channel": g,
            "B_Channel": b,
            "No_Color_Channel": no_color
        }

        # --- LINE POSITION + DASHED vs END-OF-LINE ---
        contours_line, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours_line:
            valid_contours = []
            for c in contours_line:
                area = cv2.contourArea(c)
                if area < 25:
                    continue
                rect = cv2.minAreaRect(c)
                rect_w, rect_h = rect[1]
                if rect_w == 0 or rect_h == 0:
                    continue
                min_dim = min(rect_w, rect_h)
                max_dim = max(rect_w, rect_h)

                # Line width filter: allow thin lines (3-40px)
                if not (3 <= min_dim <= 40):
                    continue

                # Extent filter: reject blobs that are too square/round (not line-like)
                extent = area / (max_dim * min_dim) if max_dim * min_dim > 0 else 0
                if extent < 0.15:
                    continue

                # Aspect ratio filter: line should be elongated
                aspect = max_dim / max(1, min_dim)
                if aspect < 1.2:
                    # Nearly square — could be a dot or intersection marker, still valid but deprioritize
                    pass

                valid_contours.append((c, area, max_dim))

            if valid_contours:
                # Sort by area, but prefer elongated contours
                valid_contours.sort(key=lambda x: x[1], reverse=True)
                largest_line = valid_contours[0][0]

                M = cv2.moments(largest_line)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]) + roi_start_x
                    cy = int(M["m01"] / M["m00"]) + roi_start_y
                    result["line_center_x"] = cx
                    result["line_center_y"] = cy

                    shifted_contour = largest_line + np.array([roi_start_x, roi_start_y])

                    # Fit line for heading vector
                    [vx, vy, x, y] = cv2.fitLine(shifted_contour, cv2.DIST_L2, 0, 0.01, 0.01)
                    vx = vx.item()
                    vy = vy.item()
                    if vy > 0:
                        vx = -vx
                        vy = -vy

                    if len(shifted_contour) >= 5:
                        ellipse = cv2.fitEllipse(shifted_contour)
                        cv2.ellipse(frame, ellipse, (255, 100, 0), 2)
                    else:
                        cv2.drawContours(frame, [shifted_contour], -1, (255, 100, 0), 2)

                    length = 70
                    pt1 = (cx, cy)
                    pt2 = (int(cx + vx * length), int(cy + vy * length))
                    cv2.arrowedLine(frame, pt1, pt2, (0, 255, 255), 4, tipLength=0.3)

                    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                    cv2.putText(frame, f"CX:{cx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # --- INTERSECTION DETECTION ---
                    x_box, y_box, w_line, h_line = cv2.boundingRect(largest_line)
                    if w_line > roi_w * 0.8:
                        result["special_state"] = "intersection"
                        cv2.putText(frame, "INTERSECTION", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            elif self._try_wide_roi_fallback(frame, result):
                pass
            else:
                result["line_center_x"] = None
                result["line_center_y"] = None

                # --- DASHED vs END-OF-LINE (improved analysis) ---
                # Use multiple scan lines across the ROI width, not just center
                scan_cols = [roi_w // 4, roi_w // 2, 3 * roi_w // 4]
                total_top = 0
                total_bottom = 0
                total_pixels = 0

                for sx in scan_cols:
                    strip = mask_black[:, max(0, sx - 3):sx + 3]
                    col_sum = np.sum(strip > 0, axis=1)
                    mid = roi_h // 2
                    total_top += np.sum(col_sum[:mid] > 0)
                    total_bottom += np.sum(col_sum[mid:] > 0)

                total_pixels = np.sum(mask_black > 0)
                black_ratio = total_pixels / (roi_h * roi_w) if roi_h * roi_w > 0 else 0

                # Also check the UNCLOSED mask for scattered pixels (dashed pattern)
                # Dashed lines have periodic black blobs; dead ends have none ahead
                row_coverage = np.sum(mask_black > 0, axis=1)
                # Find the topmost row that has any black pixels
                nonzero_rows = np.where(row_coverage > 0)[0]
                if len(nonzero_rows) > 0:
                    topmost_black_row = nonzero_rows[0]
                    bottommost_black_row = nonzero_rows[-1]
                    span = bottommost_black_row - topmost_black_row
                    span_ratio = span / roi_h if roi_h > 0 else 0
                else:
                    span_ratio = 0

                # Decision logic
                if total_top > 0 and black_ratio > 0.015:
                    # Black pixels in upper half => line continues beyond ROI => dashed gap
                    result["is_dashed"] = True
                    result["special_state"] = "dashed_gap"
                    cv2.putText(frame, "DASHED GAP", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                elif total_bottom > 0 and total_top == 0 and span_ratio < 0.5:
                    # Black only at bottom, small span => line truly ends
                    result["line_ended"] = True
                    result["special_state"] = "dead_end"
                    cv2.putText(frame, "DEAD END (180)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif total_bottom > 0 and span_ratio >= 0.5:
                    # Decent vertical span but nothing at top — likely end of line
                    result["line_ended"] = True
                    result["special_state"] = "dead_end"
                    cv2.putText(frame, "DEAD END (180)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    # Very few or no black pixels — assume dashed gap (safer: keep driving)
                    if black_ratio > 0.003:
                        result["is_dashed"] = True
                        result["special_state"] = "dashed_gap"
                        cv2.putText(frame, "DASHED GAP", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    else:
                        result["line_ended"] = True
                        result["special_state"] = "dead_end"
                        cv2.putText(frame, "DEAD END (180)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif self._try_wide_roi_fallback(frame, result):
            pass
        else:
            result["line_ended"] = True
            result["special_state"] = "dead_end"
            cv2.putText(frame, "DEAD END (180)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return result
