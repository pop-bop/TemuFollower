#!/usr/bin/env python3
import time
from collections import deque

import cv2

from config import (
    KP, KI, KD, TURN_LIMIT, CENTER_DEADZONE,
    SHARP_TURN_SPEED, MAX_TURN_SPEED, STEER_INVERT,
    BASE_SPEED, MAX_SPEED, MIN_SPEED, SPIN_SEARCH_SPEED, APPROACH_SPEED,
    LINE_LOST_STOP_TIMEOUT_S, SLEW_RATE_PER_S, MANUAL_SPEED, MANUAL_KEY_TIMEOUT_S,
    SHOW_DEBUG_VIEW, DEBUG_VIEW_EVERY_N_FRAMES, LOOP_LOG_INTERVAL_S,
    INTEGRAL_LIMIT, LOW_CONFIDENCE_SPEED_SCALE,
    ERROR_SPEED_REDUCTION, DERIVATIVE_SPEED_REDUCTION, STRAIGHT_SPEED_BOOST,
    ADAPTIVE_DERIVATIVE_REF, CURVE_SPEED_REDUCTION, CURVE_TURN_BOOST,
    MIN_CURVE_SPEED_SCALE, NEAR_TRAJECTORY_WEIGHT, LOOKAHEAD_TRAJECTORY_WEIGHT,
    LOOKAHEAD_CONFIDENCE_MIN, FAR_CONFIDENCE_MIN, CURVATURE_DAMPING,
    DERIVATIVE_SMOOTHING, ACCEL_BUZZER_ENABLED, ACCEL_SPEED_DELTA_THRESHOLD,
    MARKER_ACTION_DELAY_S,
    BACKTRACK_SPEED, ROTATE_SPEED, BACKTRACK_SEARCH_TIMEOUT_S,
    INTERSECTION_COOLDOWN_S, INTERSECTION_MEMORY_MAX_AGE_S,
    MAX_WAYPOINTS, ROI_X_START_RATIO, ROI_X_END_RATIO, ROTATE_SETTLE_TIME_S,
    VISION_DELAY_S, TURN_SLEW_RATE_PER_S,
    VISION_DELAY_MIN_S, VISION_DELAY_MAX_S, VISION_DELAY_FROM_ODOMETRY,
    ODOMETRY_ENABLED, BEV_ENABLED, BEV_NEAR_MM, BEV_CAMERA_AXLE_OFFSET_MM,
    OBSTACLE_BAND_ENABLED, OBSTACLE_BAND_MIN_FRAMES, OBSTACLE_DROPOUT_FRAMES,
    OBSTACLE_DROPOUT_MAX_RANGE_MM, OBSTACLE_TRIGGER_ON_DROPOUT,
    OBSTACLE_LATERAL_OFFSET_MM, OBSTACLE_PASS_FORWARD_MM, OBSTACLE_REACQUIRE_MM,
    OBSTACLE_MANEUVER_SPEED, OBSTACLE_MANEUVER_TURN, OBSTACLE_LEG_TIMEOUT_S,
)
from camera import (
    open_camera, read_frame, read_frame_with_motion, has_imu, get_intrinsics,
)
from vision import (
    find_line_error_normal, find_line_error_lookahead, find_line_error_far,
    find_line_error_wide, print_calibration_info, detect_intersection_normal,
    detect_obstacle_band,
)
from pi_motors import create_motors
from vision import get_warp_matrix, warp_frame, get_expected_depth_map, get_obstacle_mask
from camera import get_depth_scale
from config import CAMERA_TILT_ANGLE_DEG, CAMERA_MOUNT_HEIGHT_MM, OBSTACLE_DETECTION_ENABLED

def slew_toward(current, target, max_step):
    if target > current: return min(target, current + max_step)
    if target < current: return max(target, current - max_step)
    return current
from utils import clamp
from debug_view import draw_debug_view, draw_roi_arrow_view
from buffer import TemporalBuffer
from adaptive_pid import schedule_pid_gains
from trajectory import (
    new_trajectory_state, reset_trajectory_state, rk4_step, rk4_step_curvature,
)
import bev
from odometry import SpeedEstimator


def odometry_vision_delay(speed_mps):
    """How long ground under the near ROI takes to reach the drive wheels.

    Returns None when speed is too low to divide by, so the caller keeps the
    fixed fallback. The drive wheels are at the FRONT of the chassis, so the
    distance to cover is the near-ROI edge less any lens-ahead-of-axle offset.
    """
    if speed_mps < 0.05:
        return None
    distance_mm = max(0.0, BEV_NEAR_MM - BEV_CAMERA_AXLE_OFFSET_MM)
    return clamp(
        (distance_mm / 1000.0) / speed_mps,
        VISION_DELAY_MIN_S, VISION_DELAY_MAX_S,
    )


class ObstacleTracker:
    """Debounces band detections and decides when the robot is at standoff.

    The D435 cannot measure closer than ~105-280mm depending on preset, so depth
    inside the band drops to zero right around the standoff we want. That
    dropout IS the trigger: once a band we were already tracking at close range
    loses all valid depth, we are at the sensor's near limit. The exact standoff
    is therefore preset-dependent rather than a precise 80mm -- the trade for not
    having to dead-reckon blindly through the last few centimetres.

    The range guard matters: without it, a band lost to ordinary depth noise at
    400mm would fire the manoeuvre in open space.
    """

    def __init__(self):
        self.present_frames = 0
        self.dropout_frames = 0
        self.last_range_mm = None
        self.center_offset = 0.0
        self.prefer_left = None

    def reset(self):
        self.present_frames = 0
        self.dropout_frames = 0
        self.last_range_mm = None
        self.prefer_left = None

    def update(self, band):
        """Feed one detection. Returns True when the standoff is reached."""
        if band["found"]:
            self.present_frames += 1
            self.dropout_frames = 0
            self.last_range_mm = band["range_mm"]
            self.center_offset = band["center_offset"]
            # Pass toward whichever side has the more distant nearest
            # obstruction. Latched on first confident sight, because once the
            # obstacle fills the frame these clearances stop being meaningful.
            if self.prefer_left is None and self.present_frames >= OBSTACLE_BAND_MIN_FRAMES:
                left = band["left_clearance_mm"]
                right = band["right_clearance_mm"]
                if left is not None and right is not None and left != right:
                    self.prefer_left = left > right
                else:
                    # Nothing to choose between: go against the obstacle's own
                    # offset, which is the shorter way around.
                    self.prefer_left = band["center_offset"] > 0
            return False

        tracked = self.present_frames >= OBSTACLE_BAND_MIN_FRAMES
        close = (self.last_range_mm is not None
                 and self.last_range_mm <= OBSTACLE_DROPOUT_MAX_RANGE_MM)
        if not (tracked and close and OBSTACLE_TRIGGER_ON_DROPOUT):
            # Never got close enough to be believable -- decay the track.
            if not band["any_valid_depth"]:
                return False
            self.present_frames = max(0, self.present_frames - 1)
            if self.present_frames == 0:
                self.reset()
            return False

        self.dropout_frames += 1
        return self.dropout_frames >= OBSTACLE_DROPOUT_FRAMES


def obstacle_leg_done(estimator, mark, target_mm, started_at, now):
    """True once a manoeuvre leg has run its distance, or timed out.

    The timeout is the safety net: if flow dies on a featureless floor the
    odometer stops advancing and the leg would otherwise never finish.
    """
    if estimator is not None and estimator.travelled_since(mark) >= target_mm:
        return True
    return (now - started_at) >= OBSTACLE_LEG_TIMEOUT_S


def main():
    camera_kind, camera = open_camera()
    depth_scale = get_depth_scale(camera)
    motors, close_motors = create_motors()
    warp_M = get_warp_matrix()
    expected_depth_map = None

    first_color, first_depth = read_frame(camera_kind, camera)
    if first_color is not None:
        h, w = first_color.shape[:2]
        expected_depth_map = get_expected_depth_map(w, h, CAMERA_TILT_ANGLE_DEG, CAMERA_MOUNT_HEIGHT_MM)
        warped = warp_frame(first_color, warp_M)
        print_calibration_info(warped)

    # Bird's-eye rectification gives the ground patch a known mm/px scale, which
    # is what makes optical flow a metric speed measurement rather than a
    # scale-ambiguous one.
    bev_M = None
    bev_size = None
    bev_mask = None
    if BEV_ENABLED and first_color is not None:
        bev_M, bev_size = bev.load_bev_matrix(get_intrinsics(camera))
        if bev_M is not None:
            bev_mask = bev.bev_validity_mask(bev_M, bev_size, first_color.shape)

    speed_estimator = SpeedEstimator() if ODOMETRY_ENABLED else None
    imu_available = has_imu(camera)
    odometry_ready = speed_estimator is not None and bev_M is not None
    if ODOMETRY_ENABLED and not odometry_ready:
        print("Odometry: BEV unavailable, falling back to fixed vision delay.")
    vision_delay = VISION_DELAY_S
    last_bev_time = None

    obstacle_tracker = ObstacleTracker()
    obstacle_leg = None
    obstacle_leg_mark = 0.0
    obstacle_leg_start = 0.0
    obstacle_pass_left = True
    obstacle_retried = False

    integral = 0.0
    last_error = 0.0
    last_time = time.perf_counter()
    applied_forward = 0.0
    applied_turn = 0.0
    state = "FOLLOW"
    spin_start_time = None
    search_direction = 1.0
    mode = "AUTO"
    last_manual_key_time = time.perf_counter()
    last_manual_command = None

    prev_state = None
    prev_sharp_turn = False
    prev_red_marker = False
    prev_green_marker = False
    pending_marker_color = None
    pending_marker_action_time = None
    halted = False
    buffer = TemporalBuffer(max_size=MAX_WAYPOINTS)
    backtrack_intersection_wp = None
    backtrack_target_branch_idx = None
    backtrack_start_time = None
    rotate_settle_until = None
    last_intersection_time = 0.0
    dead_end_recorded = False
    frame_count = 0
    last_log_time = time.perf_counter()
    fps_ema = 0.0
    current_kp = KP
    current_ki = KI
    current_kd = KD
    trajectory_state = new_trajectory_state()
    curvature_state = new_trajectory_state()
    vision_delay_queue = deque()
    smoothed_derivative = 0.0
    curve_sharpness = 0.0
    curve_speed_scale = 1.0
    prev_speed_for_accel = 0.0

    print("PID line following running. Press Ctrl+C to stop.")
    print("Manual indicator test keys: r=toggle red LED, g=toggle green LED, b=buzzer blip")
    print(f"Starting speed: BASE_SPEED={BASE_SPEED}  MAX_SPEED={MAX_SPEED}  MIN_SPEED={MIN_SPEED}")

    try:
        while True:
            now = time.perf_counter()
            dt = max(0.001, now - last_time)
            last_time = now
            max_step = SLEW_RATE_PER_S * dt
            max_turn_step = TURN_SLEW_RATE_PER_S * dt
            frame_count += 1
            instant_fps = 1.0 / dt
            fps_ema = instant_fps if fps_ema <= 0.0 else (fps_ema * 0.9) + (instant_fps * 0.1)

            key = -1
            if SHOW_DEBUG_VIEW:
                key = cv2.waitKey(1) & 0xFF
                if key == 32:
                    if mode != "MANUAL":
                        print("SWITCHED TO MANUAL: use w/a/s/d, space=stop, enter=back to auto")
                    mode = "MANUAL"
                    applied_forward = 0.0
                    applied_turn = 0.0
                    last_manual_command = None
                    pending_marker_color = None
                    pending_marker_action_time = None
                    backtrack_intersection_wp = None
                    backtrack_target_branch_idx = None
                    backtrack_start_time = None
                    rotate_settle_until = None
                    dead_end_recorded = False
                    reset_trajectory_state(trajectory_state)
                    reset_trajectory_state(curvature_state)
                    vision_delay_queue.clear()
                    smoothed_derivative = 0.0
                    curve_sharpness = 0.0
                    curve_speed_scale = 1.0
                    prev_speed_for_accel = 0.0
                    motors.stop()
                elif key in (13, 10):
                    if mode != "AUTO":
                        print("SWITCHED TO AUTO: resuming line following")
                    mode = "AUTO"
                    state = "FOLLOW"
                    applied_forward = 0.0
                    applied_turn = 0.0
                    integral = 0.0
                    last_error = 0.0
                    pending_marker_color = None
                    pending_marker_action_time = None
                    backtrack_intersection_wp = None
                    backtrack_target_branch_idx = None
                    backtrack_start_time = None
                    rotate_settle_until = None
                    dead_end_recorded = False
                    spin_start_time = None
                    reset_trajectory_state(trajectory_state)
                    reset_trajectory_state(curvature_state)
                    vision_delay_queue.clear()
                    smoothed_derivative = 0.0
                    curve_sharpness = 0.0
                    curve_speed_scale = 1.0
                    prev_speed_for_accel = 0.0
                    motors.stop()
                    if halted:
                        print("MANUAL OVERRIDE: clearing red-marker halt")
                    halted = False
                elif mode == "MANUAL" and key in (ord("w"), ord("a"), ord("s"), ord("d")):
                    last_manual_key_time = now
                    last_manual_command = key
                elif key == ord("r"):
                    pass
                    print("RED LED SPI trigger (0.5s)")
                elif key == ord("g"):
                    pass
                    print("GREEN LED SPI trigger (0.5s)")
                elif key == ord("b"):
                    pass
                    print("BUZZER manual test tone")

            color_frame, depth_frame, motion_samples = read_frame_with_motion(
                camera_kind, camera
            )
            if color_frame is None:
                continue

            if speed_estimator is not None:
                # IMU samples carry their own device timestamps because the
                # D435i motion stream is not hardware-synced to the frames.
                for kind, xyz, ts in motion_samples:
                    if kind == "accel":
                        speed_estimator.add_imu_accel(xyz, ts)
                    else:
                        speed_estimator.add_imu_gyro(xyz)

                if bev_M is not None:
                    bev_gray = cv2.cvtColor(
                        bev.warp_to_bev(color_frame, bev_M, bev_size), cv2.COLOR_BGR2GRAY
                    )
                    bev_dt = 0.0 if last_bev_time is None else now - last_bev_time
                    last_bev_time = now
                    speed_estimator.update_vision(bev_gray, bev_dt, bev_mask)

                if not imu_available and speed_estimator.confidence <= 0.0:
                    # No IMU and no usable flow: nothing measures speed, so fall
                    # back to the commanded duty rather than trusting a stale value.
                    speed_estimator.fallback_from_command(applied_forward)

                # Only derive the delay from a genuinely MEASURED speed. The
                # command fallback above is just the commanded duty played back,
                # so feeding it here closed a loop (duty -> "speed" -> phase lag
                # -> duty) and swung the delay 120-300ms frame to frame, which is
                # the oscillation VISION_DELAY_* was introduced to kill.
                if (VISION_DELAY_FROM_ODOMETRY and odometry_ready
                        and speed_estimator.source == "fused"):
                    derived = odometry_vision_delay(speed_estimator.speed_mps)
                    vision_delay = VISION_DELAY_S if derived is None else derived
                else:
                    vision_delay = VISION_DELAY_S

                # Advance the odometer so manoeuvre legs can be measured in mm.
                speed_estimator.advance(dt)

            frame = warp_frame(color_frame, warp_M)
            obstacle_mask = None
            if OBSTACLE_DETECTION_ENABLED and expected_depth_map is not None and depth_frame is not None:
                obstacle_mask = get_obstacle_mask(
                    warp_frame(depth_frame, warp_M), depth_scale, expected_depth_map,
                )

            # Obstacle band is measured on the RAW depth frame, not the warped
            # one: expected_depth_map is derived from unwarped camera geometry.
            obstacle_at_standoff = False
            obstacle_band = None
            if (OBSTACLE_BAND_ENABLED and depth_frame is not None
                    and expected_depth_map is not None
                    and not state.startswith("OBSTACLE")):
                obstacle_band = detect_obstacle_band(
                    depth_frame, depth_scale, expected_depth_map,
                )
                obstacle_at_standoff = obstacle_tracker.update(obstacle_band)

            if halted:
                motors.stop()
                prev_speed_for_accel = 0.0
                if SHOW_DEBUG_VIEW:
                    cv2.imshow("Camera + Decisions", frame)
                continue

            if mode == "MANUAL":
                target_forward = 0.0
                target_turn = 0.0

                if now - last_manual_key_time < MANUAL_KEY_TIMEOUT_S and last_manual_command is not None:
                    if last_manual_command == ord("w"):
                        target_forward = MANUAL_SPEED
                        state = "MANUAL FORWARD"
                    elif last_manual_command == ord("s"):
                        target_forward = -MANUAL_SPEED
                        state = "MANUAL BACKWARD"
                    elif last_manual_command == ord("a"):
                        target_turn = -MANUAL_SPEED if STEER_INVERT else MANUAL_SPEED
                        state = "MANUAL LEFT"
                    elif last_manual_command == ord("d"):
                        target_turn = MANUAL_SPEED if STEER_INVERT else -MANUAL_SPEED
                        state = "MANUAL RIGHT"
                    else:
                        state = "MANUAL IDLE"
                else:
                    state = "MANUAL IDLE"

                applied_forward = slew_toward(applied_forward, target_forward, max_step)
                turn_component = slew_toward(
                    applied_turn,
                    clamp(target_turn, -MAX_TURN_SPEED, MAX_TURN_SPEED),
                    max_turn_step,
                )
                applied_turn = turn_component
                applied_left = clamp(applied_forward + turn_component, -1.0, 1.0)
                applied_right = clamp(applied_forward - turn_component, -1.0, 1.0)

                motors.set_speeds(applied_left, applied_right)

                current_speed_mag = abs(applied_forward)
                accelerating = ACCEL_BUZZER_ENABLED and current_speed_mag > prev_speed_for_accel + ACCEL_SPEED_DELTA_THRESHOLD
                prev_speed_for_accel = current_speed_mag

                if accelerating:
                    pass
                prev_state = state

                _, active_debug = find_line_error_normal(frame, obstacle_mask)
                if SHOW_DEBUG_VIEW and frame_count % DEBUG_VIEW_EVERY_N_FRAMES == 0:
                    draw_debug_view(frame, active_debug, None, None, applied_left, applied_right, state)
                    draw_roi_arrow_view(active_debug, applied_left, applied_right, state)

                if now - last_log_time >= LOOP_LOG_INTERVAL_S:
                    last_log_time = now
                    print(f"state={state:16s} applied L/R={applied_left:+.2f}/{applied_right:+.2f} fps={fps_ema:.1f}")
                continue

            # ===== AUTO MODE: VISION + STATE MACHINE =====
            normal_error, normal_debug = find_line_error_normal(frame, obstacle_mask)

            target_forward = 0.0
            target_turn = 0.0
            display_turn = None
            active_debug = normal_debug
            lookahead_debug = None
            far_debug = None
            curve_sharpness = 0.0
            curve_speed_scale = 1.0

            # INTERSECTION DETECTION (only when centered on the line)
            if (state not in ("BACKTRACK", "ROTATE")
                    and not state.startswith("OBSTACLE")
                    and normal_error is not None
                    and abs(normal_error) < CENTER_DEADZONE * 2):
                if now - last_intersection_time > INTERSECTION_COOLDOWN_S:
                    intersection = detect_intersection_normal(frame, obstacle_mask)
                    if intersection:
                        inter_wp = buffer.record(
                            frame=frame, state=state, error=normal_error,
                            is_intersection=True,
                            branches=intersection["branches"],
                            chosen_branch_idx=intersection["chosen_branch_idx"],
                        )
                        last_intersection_time = now
                        print(f"INTERSECTION: {intersection['branch_count']} branches, "
                              f"chosen idx {inter_wp['chosen_branch_idx']}")

            # ----- OBSTACLE GO-AROUND -----
            # Entered when a tracked band's depth drops out, i.e. we are at the
            # sensor's near limit. Legs are dead-reckoned on fused odometry
            # because the obstacle is inside the depth blind zone throughout:
            # there is nothing left to measure against once we are this close.
            if obstacle_at_standoff and not state.startswith("OBSTACLE"):
                obstacle_pass_left = (obstacle_tracker.prefer_left
                                      if obstacle_tracker.prefer_left is not None else True)
                obstacle_retried = False
                obstacle_leg = "CLEAR"
                obstacle_leg_mark = speed_estimator.mark() if speed_estimator else 0.0
                obstacle_leg_start = now
                state = "OBSTACLE_CLEAR"
                motors.stop()
                print(f"OBSTACLE at ~{obstacle_tracker.last_range_mm:.0f}mm "
                      f"(depth dropout) -- passing "
                      f"{'LEFT' if obstacle_pass_left else 'RIGHT'}")
                continue

            if state.startswith("OBSTACLE"):
                # Steering sign: positive turn goes one way, negated under
                # STEER_INVERT exactly as everywhere else in this loop.
                away = -1.0 if obstacle_pass_left else 1.0
                leg_turn = OBSTACLE_MANEUVER_TURN * away
                if STEER_INVERT:
                    leg_turn = -leg_turn

                if obstacle_leg == "CLEAR":
                    # Arc away from the line until clear of the obstacle's width.
                    target_forward = OBSTACLE_MANEUVER_SPEED
                    target_turn = leg_turn
                    display_turn = leg_turn
                    if obstacle_leg_done(speed_estimator, obstacle_leg_mark,
                                         OBSTACLE_LATERAL_OFFSET_MM,
                                         obstacle_leg_start, now):
                        obstacle_leg = "PASS"
                        obstacle_leg_mark = speed_estimator.mark() if speed_estimator else 0.0
                        obstacle_leg_start = now
                        state = "OBSTACLE_PASS"

                elif obstacle_leg == "PASS":
                    # Run straight past the obstacle.
                    target_forward = OBSTACLE_MANEUVER_SPEED
                    target_turn = 0.0
                    if obstacle_leg_done(speed_estimator, obstacle_leg_mark,
                                         OBSTACLE_PASS_FORWARD_MM,
                                         obstacle_leg_start, now):
                        obstacle_leg = "RETURN"
                        obstacle_leg_mark = speed_estimator.mark() if speed_estimator else 0.0
                        obstacle_leg_start = now
                        state = "OBSTACLE_RETURN"

                elif obstacle_leg == "RETURN":
                    # Cut back toward the line, watching for it the whole way so
                    # a short obstacle does not overshoot the return.
                    target_forward = OBSTACLE_MANEUVER_SPEED
                    target_turn = -leg_turn
                    display_turn = -leg_turn
                    if normal_error is not None:
                        print("OBSTACLE: line reacquired")
                        obstacle_tracker.reset()
                        obstacle_leg = None
                        state = "FOLLOW"
                        vision_delay_queue.clear()
                        reset_trajectory_state(trajectory_state)
                        reset_trajectory_state(curvature_state)
                        integral = 0.0
                        continue
                    if obstacle_leg_done(speed_estimator, obstacle_leg_mark,
                                         OBSTACLE_LATERAL_OFFSET_MM,
                                         obstacle_leg_start, now):
                        obstacle_leg = "REACQUIRE"
                        obstacle_leg_mark = speed_estimator.mark() if speed_estimator else 0.0
                        obstacle_leg_start = now
                        state = "OBSTACLE_REACQUIRE"

                elif obstacle_leg == "REACQUIRE":
                    # Creep forward hunting for the line. The wide ROI is used
                    # because the line may still be off to one side.
                    wide_err, wide_debug = find_line_error_wide(frame, obstacle_mask)
                    active_debug = wide_debug
                    if normal_error is not None or wide_err is not None:
                        print("OBSTACLE: line reacquired (wide)")
                        obstacle_tracker.reset()
                        obstacle_leg = None
                        state = "FOLLOW"
                        vision_delay_queue.clear()
                        reset_trajectory_state(trajectory_state)
                        reset_trajectory_state(curvature_state)
                        integral = 0.0
                        continue

                    target_forward = OBSTACLE_MANEUVER_SPEED * 0.8
                    target_turn = 0.0
                    if obstacle_leg_done(speed_estimator, obstacle_leg_mark,
                                         OBSTACLE_REACQUIRE_MM,
                                         obstacle_leg_start, now):
                        if not obstacle_retried:
                            # Wrong side, or the obstacle was wider than assumed.
                            # Reverse out and try the other way around.
                            obstacle_retried = True
                            obstacle_pass_left = not obstacle_pass_left
                            obstacle_leg = "BACKOUT"
                            obstacle_leg_mark = speed_estimator.mark() if speed_estimator else 0.0
                            obstacle_leg_start = now
                            state = "OBSTACLE_BACKOUT"
                            print("OBSTACLE: no line -- backing out to retry "
                                  f"{'LEFT' if obstacle_pass_left else 'RIGHT'}")
                        else:
                            print("OBSTACLE: both sides failed -- falling back to search")
                            obstacle_tracker.reset()
                            obstacle_leg = None
                            state = "SPIN_SEARCH"
                            spin_start_time = now

                elif obstacle_leg == "BACKOUT":
                    # Retrace far enough to be beside the obstacle again before
                    # arcing the other way.
                    target_forward = -OBSTACLE_MANEUVER_SPEED
                    target_turn = 0.0
                    if obstacle_leg_done(speed_estimator, obstacle_leg_mark,
                                         OBSTACLE_PASS_FORWARD_MM + OBSTACLE_REACQUIRE_MM,
                                         obstacle_leg_start, now):
                        obstacle_leg = "CLEAR"
                        obstacle_leg_mark = speed_estimator.mark() if speed_estimator else 0.0
                        obstacle_leg_start = now
                        state = "OBSTACLE_CLEAR"

            # ----- BACKTRACK: reverse along the line to reach last intersection -----
            elif state == "BACKTRACK":
                backtrack_err, backtrack_debug = find_line_error_normal(frame, obstacle_mask)
                active_debug = backtrack_debug

                inter = detect_intersection_normal(frame, obstacle_mask)
                if inter:
                    print(f"BACKTRACK: reached intersection ({inter['branch_count']} branches)")
                    motors.stop()
                    backtrack_start_time = None
                    rotate_settle_until = now + ROTATE_SETTLE_TIME_S
                    state = "ROTATE"
                    continue

                if backtrack_err is not None:
                    error = backtrack_err
                    raw_turn = KP * error
                    raw_turn = clamp(raw_turn, -TURN_LIMIT, TURN_LIMIT)
                    display_turn = raw_turn
                    motor_turn = -raw_turn if STEER_INVERT else raw_turn
                    target_forward = -BACKTRACK_SPEED
                    target_turn = motor_turn
                else:
                    wide_err, wide_debug = find_line_error_wide(frame, obstacle_mask)
                    active_debug = wide_debug
                    if wide_err is not None:
                        raw_turn = clamp(KP * wide_err, -TURN_LIMIT, TURN_LIMIT)
                        display_turn = raw_turn
                        motor_turn = -raw_turn if STEER_INVERT else raw_turn
                        target_forward = -APPROACH_SPEED
                        target_turn = motor_turn
                    else:
                        if backtrack_start_time and now - backtrack_start_time > BACKTRACK_SEARCH_TIMEOUT_S:
                            print("BACKTRACK FAILED: intersection not found")
                            motors.stop()
                            state = "STOP"
                        else:
                            target_forward = 0.0
                            target_turn = SPIN_SEARCH_SPEED * search_direction

            # ----- ROTATE: turn toward the untried branch at the intersection -----
            elif state == "ROTATE":
                if rotate_settle_until is not None and now < rotate_settle_until:
                    target_forward = 0.0
                    target_turn = 0.0
                    display_turn = 0.0
                    active_debug = normal_debug
                    prev_green_marker = False
                    prev_red_marker = False
                else:
                    rotate_settle_until = None

                    if backtrack_start_time is None:
                        backtrack_start_time = now
                    rotate_elapsed = now - backtrack_start_time
                    if rotate_elapsed > BACKTRACK_SEARCH_TIMEOUT_S:
                        print("ROTATE TIMEOUT: giving up")
                        backtrack_intersection_wp = None
                        backtrack_target_branch_idx = None
                        backtrack_start_time = None
                        rotate_settle_until = None
                        state = "STOP"
                    else:
                        inter = detect_intersection_normal(frame, obstacle_mask)
                        if inter and backtrack_target_branch_idx is not None and backtrack_target_branch_idx < len(inter["branches"]):
                            target_branch = inter["branches"][backtrack_target_branch_idx]
                            roi_w = frame.shape[1] * (ROI_X_END_RATIO - ROI_X_START_RATIO)
                            branch_error = (target_branch["cx"] - inter["roi_center_x"]) / max(1.0, roi_w / 2.0)

                            if abs(branch_error) < CENTER_DEADZONE * 0.5:
                                print("ROTATE: branch centered, switching to FOLLOW")
                                if backtrack_intersection_wp is not None:
                                    buffer.mark_branch_taken(backtrack_intersection_wp, backtrack_target_branch_idx)
                                    buffer.clear_after(backtrack_intersection_wp["timestamp"])
                                backtrack_intersection_wp = None
                                backtrack_target_branch_idx = None
                                backtrack_start_time = None
                                rotate_settle_until = None
                                dead_end_recorded = False
                                integral = 0.0
                                last_error = 0.0
                                search_direction = 1.0
                                reset_trajectory_state(trajectory_state)
                                reset_trajectory_state(curvature_state)
                                vision_delay_queue.clear()
                                smoothed_derivative = 0.0
                                state = "FOLLOW"
                            else:
                                turn = clamp(branch_error * ROTATE_SPEED * 2, -ROTATE_SPEED, ROTATE_SPEED)
                                if STEER_INVERT:
                                    turn = -turn
                                target_forward = 0.0
                                target_turn = turn
                                display_turn = turn
                        else:
                            target_forward = 0.0
                            target_turn = ROTATE_SPEED * 0.5 if not STEER_INVERT else -ROTATE_SPEED * 0.5

            # ----- NORMAL LINE FOLLOWING -----
            elif normal_error is not None:
                state = "FOLLOW"
                dead_end_recorded = False

                # Look ahead above the normal ROI to anticipate curves before
                # the near ROI reacts to them.
                lookahead_error, lookahead_debug = find_line_error_lookahead(frame, obstacle_mask)
                lookahead_confidence = lookahead_debug.get("line_confidence", 0.0)
                lookahead_ok = lookahead_error is not None and lookahead_confidence >= LOOKAHEAD_CONFIDENCE_MIN

                # Third ROI, farther out still -- lets the robot see a curve
                # closing back out before the lookahead ROI does.
                far_error, far_debug = find_line_error_far(frame, obstacle_mask)
                far_confidence = far_debug.get("line_confidence", 0.0)
                far_ok = far_error is not None and far_confidence >= FAR_CONFIDENCE_MIN

                if lookahead_ok:
                    curve = lookahead_error - normal_error
                    # Trust the look-ahead ROI more at speed and when already
                    # centered; fall back to the near ROI when off-center or
                    # nearly stopped, since the near line is the safety anchor.
                    speed_factor = clamp(abs(applied_forward) / max(BASE_SPEED, 0.001), 0.0, 1.0)
                    off_center_factor = clamp(abs(normal_error) / max(CENTER_DEADZONE, 0.001), 0.0, 1.0)
                    lookahead_trust = speed_factor * (1.0 - 0.5 * off_center_factor)
                    lookahead_weight = LOOKAHEAD_TRAJECTORY_WEIGHT * lookahead_trust
                    near_weight = NEAR_TRAJECTORY_WEIGHT + LOOKAHEAD_TRAJECTORY_WEIGHT * (1.0 - lookahead_trust)
                    trajectory_target_error = (near_weight * normal_error) + (lookahead_weight * lookahead_error)
                else:
                    curve = 0.0
                    trajectory_target_error = normal_error

                predicted_error, predicted_heading = rk4_step(
                    trajectory_state, trajectory_target_error, curve, dt,
                )

                # ----- Layer 2: curvature trend from the lookahead/far span -----
                # far_curve is the curvature the near/lookahead layer will see a
                # moment from now; comparing it against the current curve tells
                # us whether the turn is tightening or closing back out.
                far_curve = (far_error - lookahead_error) if (lookahead_ok and far_ok) else curve
                curvature_rate_target = far_curve - curve
                predicted_curvature, predicted_curvature_rate = rk4_step_curvature(
                    curvature_state, far_curve, curvature_rate_target, dt,
                )

                curve_sharpness = clamp(abs(predicted_heading), 0.0, 1.0)
                # If the curvature rate opposes the current heading, the turn is
                # closing out -- damp the turn boost pre-emptively instead of
                # waiting for the near ROI to straighten and overshooting the
                # correction (a common cause of oscillation on exits).
                # Applied continuously rather than as a sign test: a binary switch
                # on the product of two noisy estimates toggled at frame rate.
                closing_amount = clamp(
                    -predicted_curvature_rate * predicted_heading, 0.0, 1.0,
                )
                curvature_damping = clamp(1.0 - CURVATURE_DAMPING * closing_amount, 0.0, 1.0)

                curve_speed_scale = clamp(
                    1.0 - CURVE_SPEED_REDUCTION * curve_sharpness,
                    MIN_CURVE_SPEED_SCALE, 1.0,
                )

                future_error = predicted_error
                raw_derivative = (future_error - last_error) / dt
                # Smooth the derivative -- a raw frame-to-frame derivative of a
                # noisy vision error is the classic driver of turn oscillation.
                smoothed_derivative = (
                    (1.0 - DERIVATIVE_SMOOTHING) * raw_derivative
                    + DERIVATIVE_SMOOTHING * smoothed_derivative
                )
                derivative = smoothed_derivative
                last_error = future_error

                # Persistence of vision: the camera is tilted forward, so the ROI
                # centroid is ground the wheels only reach vision_delay later.
                # An earlier version derived this from a single raw depth pixel
                # inside the sensor's blind zone and swung the loop's phase lag
                # 141-450ms frame to frame, which oscillates at any gain. It is
                # now derived from FUSED speed, which is smooth and clamped to
                # [VISION_DELAY_MIN_S, VISION_DELAY_MAX_S], and falls back to the
                # fixed VISION_DELAY_S whenever odometry is unhealthy.
                vision_delay_queue.append((now, future_error))
                while len(vision_delay_queue) > 1 and (now - vision_delay_queue[1][0]) >= vision_delay:
                    vision_delay_queue.popleft()
                error = clamp(vision_delay_queue[0][1], -1.0, 1.0)
                
                integral += error * dt
                integral = clamp(integral, -INTEGRAL_LIMIT, INTEGRAL_LIMIT)

                line_confidence = normal_debug.get("line_confidence", 1.0)
                current_kp, current_ki, current_kd = schedule_pid_gains(
                    KP, KI, KD, error, derivative, line_confidence,
                )

                raw_turn = (current_kp * error) + (current_ki * integral) + (current_kd * derivative)
                raw_turn *= 1.0 + CURVE_TURN_BOOST * curve_sharpness * curvature_damping
                raw_turn = clamp(raw_turn, -TURN_LIMIT, TURN_LIMIT)
                display_turn = raw_turn
                motor_turn = -raw_turn if STEER_INVERT else raw_turn

                abs_error = abs(error)
                if abs_error <= CENTER_DEADZONE:
                    straight_bonus = STRAIGHT_SPEED_BOOST * (1.0 - abs_error / max(CENTER_DEADZONE, 0.001))
                    target_forward = BASE_SPEED + straight_bonus
                    target_turn = motor_turn
                else:
                    blend = clamp(
                        (abs_error - CENTER_DEADZONE) / max(0.001, 1.0 - CENTER_DEADZONE),
                        0.0, 1.0,
                    )
                    turn_sign = 1.0 if motor_turn >= 0 else -1.0
                    target_forward = BASE_SPEED * (1.0 - blend)
                    target_turn = motor_turn * (1.0 - blend) + (SHARP_TURN_SPEED * turn_sign) * blend

                # The blend above deliberately drives forward speed to zero on the
                # sharpest turns so the robot can pivot. Remember whether it ever
                # intended to translate, because the clamp below must not
                # resurrect a speed the geometry just cancelled.
                wants_to_translate = target_forward >= MIN_SPEED

                derivative_load = clamp(
                    abs(derivative) / max(0.001, ADAPTIVE_DERIVATIVE_REF),
                    0.0, 1.0,
                )
                stability_scale = 1.0 - (ERROR_SPEED_REDUCTION * abs_error)
                stability_scale -= DERIVATIVE_SPEED_REDUCTION * derivative_load
                target_forward *= clamp(stability_scale, MIN_SPEED / max(BASE_SPEED, 0.001), 1.0)
                target_forward *= curve_speed_scale

                if line_confidence < 0.5:
                    confidence_scale = LOW_CONFIDENCE_SPEED_SCALE + (
                        (1.0 - LOW_CONFIDENCE_SPEED_SCALE) * line_confidence * 2.0
                    )
                    target_forward *= clamp(confidence_scale, LOW_CONFIDENCE_SPEED_SCALE, 1.0)

                # Floor at MIN_SPEED only if we meant to move forward at all.
                # Flooring unconditionally undid every taper above and pinned
                # forward at exactly MIN_SPEED through hard turns, forcing the
                # robot to creep through corners it should have pivoted around.
                target_forward = clamp(
                    target_forward, MIN_SPEED if wants_to_translate else 0.0, MAX_SPEED,
                )

                search_direction = 1.0 if error >= 0 else -1.0
                if STEER_INVERT:
                    search_direction = -search_direction
                spin_start_time = None

                green_marker = normal_debug.get("green_marker", False)
                red_marker = normal_debug.get("red_marker", False)
                if red_marker and not prev_red_marker:
                    print(f"RED MARKER: stopping in {MARKER_ACTION_DELAY_S:.1f}s")
                    pending_marker_color = "red"
                    pending_marker_action_time = now + MARKER_ACTION_DELAY_S
                elif green_marker and not prev_green_marker and pending_marker_color is None:
                    print(f"GREEN MARKER: continuing in {MARKER_ACTION_DELAY_S:.1f}s")
                    pending_marker_color = "green"
                    pending_marker_action_time = now + MARKER_ACTION_DELAY_S
                prev_green_marker = green_marker
                prev_red_marker = red_marker

            # ----- LINE LOST: APPROACH or SPIN_SEARCH -----
            else:
                prev_green_marker = False
                prev_red_marker = False

                wide_error, wide_debug = find_line_error_wide(frame, obstacle_mask)

                if wide_error is not None:
                    state = "APPROACH"
                    dead_end_recorded = False
                    active_debug = wide_debug
                    raw_turn = clamp(KP * wide_error, -TURN_LIMIT, TURN_LIMIT)
                    display_turn = raw_turn
                    motor_turn = -raw_turn if STEER_INVERT else raw_turn
                    target_forward = APPROACH_SPEED
                    target_turn = motor_turn
                    search_direction = 1.0 if wide_error >= 0 else -1.0
                    if STEER_INVERT:
                        search_direction = -search_direction
                    spin_start_time = None
                else:
                    state = "SPIN_SEARCH"
                    active_debug = wide_debug
                    if spin_start_time is None:
                        spin_start_time = now

                    if now - spin_start_time < LINE_LOST_STOP_TIMEOUT_S:
                        target_forward = 0.0
                        # search_direction already has STEER_INVERT folded in
                        # (see where it is assigned), so it is used unnegated --
                        # matching the BACKTRACK search. Negating it here spun the
                        # robot away from where the line was last seen.
                        target_turn = SPIN_SEARCH_SPEED * search_direction
                    else:
                        # Dead end: try backtracking if we have intersection history
                        if not dead_end_recorded:
                            buffer.record_dead_end(frame=frame, state=state)
                            dead_end_recorded = True

                        if backtrack_intersection_wp is None:
                            inter_wp = buffer.find_recent_intersection(
                                now, max_age=INTERSECTION_MEMORY_MAX_AGE_S,
                            )
                            if inter_wp and buffer.has_untried_branches(inter_wp):
                                branch_idx, branch = buffer.find_untried_branch(inter_wp)
                                if branch is not None:
                                    print("DEAD END: backtracking to last intersection "
                                          f"and trying branch {branch_idx}")
                                    backtrack_intersection_wp = inter_wp
                                    backtrack_target_branch_idx = branch_idx
                                    backtrack_start_time = now
                                    rotate_settle_until = None
                                    buffer.mark_chosen_branch_taken(inter_wp)
                                    state = "BACKTRACK"
                                    target_forward = 0.0
                                    target_turn = 0.0
                                else:
                                    target_forward = 0.0
                                    target_turn = 0.0
                            else:
                                print("DEAD END: no untried intersection branch found")
                                state = "STOP"
                                target_forward = 0.0
                                target_turn = 0.0
                        else:
                            target_forward = 0.0
                            target_turn = 0.0

            if pending_marker_color is not None and now >= pending_marker_action_time:
                if pending_marker_color == "green":
                    print("GREEN MARKER: continuing")
                    pass
                elif pending_marker_color == "red":
                    print("RED MARKER: stopping")
                    pass
                    motors.stop()
                    halted = True
                pending_marker_color = None
                pending_marker_action_time = None


            if halted:
                if SHOW_DEBUG_VIEW and frame_count % DEBUG_VIEW_EVERY_N_FRAMES == 0:
                    cv2.imshow("Camera + Decisions", frame)
                prev_speed_for_accel = 0.0
                continue

            # ----- STATE TRANSITION SOUNDS -----
            if state != prev_state and state in ("SPIN_SEARCH", "APPROACH", "FOLLOW", "BACKTRACK", "ROTATE"):
                pass

            prev_state = state

            # ----- MOTOR SPEED GENERATION & TRANSMIT -----
            if state == "STOP":
                motors.stop()
                applied_forward = 0.0
                applied_turn = 0.0
                applied_left = 0.0
                applied_right = 0.0
                display_turn = 0.0
                turn_component = 0.0
            else:
                applied_forward = slew_toward(applied_forward, target_forward, max_step)

                # Update visual variables for debug UI
                turn_component = slew_toward(
                    applied_turn,
                    clamp(target_turn, -MAX_TURN_SPEED, MAX_TURN_SPEED),
                    max_turn_step,
                )
                applied_turn = turn_component
                applied_left = clamp(applied_forward + turn_component, -1.0, 1.0)
                applied_right = clamp(applied_forward - turn_component, -1.0, 1.0)
                motors.set_speeds(applied_left, applied_right)

            if SHOW_DEBUG_VIEW and frame_count % DEBUG_VIEW_EVERY_N_FRAMES == 0:
                extra_lines = None
                lookahead_bounds = None
                lookahead_point = None
                far_bounds = None
                far_point = None
                if state == "FOLLOW" and lookahead_debug is not None:
                    extra_lines = [f"curve={curve_sharpness:.2f} spd_scale={curve_speed_scale:.2f}"]
                    lookahead_bounds = lookahead_debug.get("roi_bounds")
                    lookahead_point = lookahead_debug.get("line_point")
                    if far_debug is not None:
                        far_bounds = far_debug.get("roi_bounds")
                        far_point = far_debug.get("line_point")
                draw_debug_view(
                    frame, active_debug,
                    normal_error if state == "FOLLOW" else None,
                    display_turn, applied_left, applied_right, state,
                    extra_lines=extra_lines,
                    lookahead_bounds=lookahead_bounds,
                    lookahead_point=lookahead_point,
                    far_bounds=far_bounds,
                    far_point=far_point,
                )
                lean_ratio = (display_turn / TURN_LIMIT) if display_turn is not None else None
                draw_roi_arrow_view(active_debug, applied_left, applied_right, state, lean_ratio)

            if now - last_log_time >= LOOP_LOG_INTERVAL_S:
                last_log_time = now
                loop_ms = dt * 1000.0
                if speed_estimator is not None:
                    odo = (f" v={speed_estimator.speed_mps:.2f}m/s"
                           f"({speed_estimator.source},c={speed_estimator.confidence:.2f})"
                           f" vdel={vision_delay*1000:.0f}ms")
                else:
                    odo = ""
                print(
                    f"state={state:12s} forward={applied_forward:+.2f} turn={turn_component:+.2f}  "
                    f"L/R={applied_left:+.2f}/{applied_right:+.2f}  loop_ms={loop_ms:.1f} "
                    f"fps={fps_ema:.1f} pid={current_kp:.2f}/{current_ki:.2f}/{current_kd:.2f} "
                    f"curve={curve_sharpness:.2f}/{curve_speed_scale:.2f}{odo}"
                )
    except KeyboardInterrupt:
        print("stopping")

    finally:
        motors.stop()
        close_motors()
        if camera_kind == "picamera2":
            camera.stop()
        elif camera_kind == "realsense":
            camera[0].stop()
        elif camera_kind == "usb":
            camera.release()
        else:
            if hasattr(camera, 'release'): camera.release()
        if SHOW_DEBUG_VIEW:
            cv2.destroyAllWindows()
        print("cleaned up")



if __name__ == "__main__":
    main()
