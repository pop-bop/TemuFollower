import cv2
import numpy as np

class VisionAgent:
    def __init__(self, resolution=(320, 240)):
        self.resolution = resolution
        self.use_picamera = False
        self.simulation_mode = False

        # On Raspberry Pi, try to open the camera directly via OpenCV (V4L2 backend).
        # No picamera module needed.
        print("[Vision] Opening camera via OpenCV (V4L2)...")
        self.camera = None
        for cam_index in [0, 1, 2]:
            try:
                cam = cv2.VideoCapture(cam_index)
                if cam.isOpened():
                    ret, test_frame = cam.read()
                    if ret and test_frame is not None:
                        cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                        cam.set(cv2.CAP_PROP_FPS, 30)
                        self.camera = cam
                        actual_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print(f"[Vision] Camera opened at index {cam_index} ({actual_w}x{actual_h})")
                        break
                    cam.release()
            except Exception as e:
                print(f"[Vision] Index {cam_index} failed: {e}")
                try:
                    cam.release()
                except Exception:
                    pass

        if self.camera is None:
            print("[Vision] No camera found. Running in simulation mode.")
            self.simulation_mode = True
            return

        # HOG pedestrian detector (built into OpenCV, no extra files needed)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

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
                ret, frame = self.camera.read()
                if not ret:
                    print("[Vision] Camera read failed. Reconnecting...")
                    self._reconnect_camera()
                    if self.camera is None:
                        break
                    continue
                yield frame

    def _reconnect_camera(self):
        """Try to reconnect to the camera after a failure."""
        if self.camera is not None:
            try:
                self.camera.release()
            except Exception:
                pass
        import time
        time.sleep(1.0)
        for cam_index in [0, 1, 2]:
            try:
                cam = cv2.VideoCapture(cam_index)
                if cam.isOpened():
                    ret, _ = cam.read()
                    if ret:
                        cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                        self.camera = cam
                        print(f"[Vision] Reconnected at index {cam_index}")
                        return
                cam.release()
            except Exception:
                pass
        print("[Vision] Reconnect failed. No camera available.")
        self.camera = None

    def _detect_humans(self, roi):
        """HOG pedestrian detection on the ROI.
        Returns a binary mask (same size as ROI) where white = human region.
        """
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        if roi.size == 0:
            return mask

        # HOG needs a reasonably sized image; resize ROI up for better detection
        rh, rw = roi.shape[:2]
        scale = max(1.0, 64.0 / rh)  # ensure minimum height for HOG
        if scale > 1.0:
            roi_scaled = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        else:
            roi_scaled = roi

        # Multi-scale detection with different winStride and padding
        boxes, weights = self.hog.detectMultiScale(
            roi_scaled,
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.05
        )

        # Map boxes back to original ROI coordinates and paint mask
        for (x, y, w, h) in boxes:
            # Reverse the scaling
            ox = int(x / scale)
            oy = int(y / scale)
            ow = int(w / scale)
            oh = int(h / scale)
            # Clamp to ROI bounds
            ox = max(0, ox)
            oy = max(0, oy)
            ow = min(ow, rw - ox)
            oh = min(oh, rh - oy)
            # Dilate the mask a bit to cover limbs/edges
            pad = max(5, int(ow * 0.2))
            y1 = max(0, oy - pad)
            y2 = min(rh, oy + oh + pad)
            x1 = max(0, ox - pad)
            x2 = min(rw, ox + ow + pad)
            mask[y1:y2, x1:x2] = 255

        return mask

    def _improve_line_mask(self, gray_roi, human_mask):
        """
        Multi-method black line mask with human regions excluded.
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

        # --- Method 3: Color-based (very dark pixels across all channels) ---
        b, g, r = cv2.split(blurred)
        mask_color = np.zeros_like(gray_roi)
        mask_color[(b < 70) & (g < 70) & (r < 70)] = 255

        # Combine: a pixel is "line" if ANY method says so (union)
        mask_combined = cv2.bitwise_or(mask_global, mask_adaptive)
        mask_combined = cv2.bitwise_or(mask_combined, mask_color)

        # Morphological cleanup
        kernel_open = np.ones((3, 3), np.uint8)
        mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel_open)

        # CLOSE to bridge small gaps in solid lines
        kernel_close = np.ones((5, 5), np.uint8)
        mask_closed = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel_close)

        # --- EXCLUDE HUMAN REGIONS ---
        # Where human_mask is white, paint the line mask black (exclude)
        mask_combined[human_mask > 0] = 0
        mask_closed[human_mask > 0] = 0

        return mask_combined, mask_closed, mask_global, mask_adaptive, mask_color

    def process_frame(self, frame):
        """
        Processes a single frame for line following and maze solving.
        Returns a dict containing line position, red line tracking, and special states.
        """
        result = {
            "line_center_x": None,
            "line_center_y": None,
            "red_line_detected": False,
            "red_line_center_x": None,
            "red_line_center_y": None,
            "line_ended": False,
            "is_dashed": False,
            "special_state": None,
            "cnn_layers": None,
            "human_detected": False
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

        # --- HUMAN DETECTION ---
        human_mask = self._detect_humans(roi)
        if np.any(human_mask > 0):
            result["human_detected"] = True
            # Draw human detection boxes on the main frame for debug
            contours_human, _ = cv2.findContours(human_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for ch in contours_human:
                x, y, ww, hh = cv2.boundingRect(ch)
                cv2.rectangle(frame, (x + roi_start_x, y + roi_start_y),
                              (x + roi_start_x + ww, y + roi_start_y + hh),
                              (255, 0, 255), 2)
                cv2.putText(frame, "HUMAN (MASKED)", (x + roi_start_x, y + roi_start_y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # --- RED LINE DETECTION ---
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_red_1 = cv2.inRange(hsv_roi, np.array([0, 120, 70]), np.array([10, 255, 255]))
        mask_red_2 = cv2.inRange(hsv_roi, np.array([170, 120, 70]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)

        # Exclude humans from red detection too
        mask_red[human_mask > 0] = 0

        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_red:
            largest_red = max(contours_red, key=cv2.contourArea)
            if cv2.contourArea(largest_red) > 300:
                x, y, w_box, h_box = cv2.boundingRect(largest_red)
                real_y = y + roi_start_y
                real_x = x + roi_start_x
                aspect_ratio = float(w_box) / max(1, h_box)

                result["red_line_detected"] = True
                result["red_line_center_x"] = real_x + w_box // 2
                result["red_line_center_y"] = real_y + h_box // 2

                if aspect_ratio > 2.0:
                    cv2.rectangle(frame, (real_x, real_y), (real_x + w_box, real_y + h_box), (0, 0, 255), -1)
                    cv2.putText(frame, "RED LINE (TRACK)", (real_x, real_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    if real_y > frame.shape[0] * 0.75:
                        result["special_state"] = "red_line_beneath"
                else:
                    cv2.rectangle(frame, (real_x, real_y), (real_x + w_box, real_y + h_box), (0, 0, 255), 2)
                    cv2.putText(frame, "RED TARGET", (real_x, real_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # --- IMPROVED BLACK LINE DETECTION ---
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Paint out red so it doesn't interfere with line detection
        gray_roi_for_line = gray_roi.copy()
        gray_roi_for_line[mask_red > 0] = 255

        mask_black, mask_closed, mask_global, mask_adaptive, mask_color = \
            self._improve_line_mask(gray_roi_for_line, human_mask)

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
        else:
            result["line_ended"] = True
            result["special_state"] = "dead_end"
            cv2.putText(frame, "DEAD END (180)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return result
