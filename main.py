#!/usr/bin/env python3
import time

import cv2
import RPi.GPIO as GPIO

from config import (
    KP, KI, KD, TURN_LIMIT, CENTER_DEADZONE,
    SHARP_TURN_SPEED, MAX_TURN_SPEED, STEER_INVERT,
    BASE_SPEED, MAX_SPEED, MIN_SPEED, SPIN_SEARCH_SPEED, APPROACH_SPEED,
    LINE_LOST_STOP_TIMEOUT_S, SLEW_RATE_PER_S, MANUAL_SPEED, MANUAL_KEY_TIMEOUT_S,
    SHOW_DEBUG_VIEW, DEBUG_VIEW_EVERY_N_FRAMES, LOOP_LOG_INTERVAL_S,
    INTEGRAL_LIMIT, LOW_CONFIDENCE_SPEED_SCALE,
    ERROR_SPEED_REDUCTION, DERIVATIVE_SPEED_REDUCTION, STRAIGHT_SPEED_BOOST,
    ADAPTIVE_DERIVATIVE_REF,
    MARKER_ACTION_DELAY_S,
    RED_LED_PIN, GREEN_LED_PIN,
    BACKTRACK_SPEED, ROTATE_SPEED, BACKTRACK_SEARCH_TIMEOUT_S,
    INTERSECTION_COOLDOWN_S, INTERSECTION_MEMORY_MAX_AGE_S,
    MAX_WAYPOINTS, ROI_X_START_RATIO, ROI_X_END_RATIO, ROTATE_SETTLE_TIME_S,
)
from camera import open_camera, read_frame
from vision import (
    find_line_error_normal, find_line_error_wide, print_calibration_info,
    detect_intersection_normal,
)
from motors import setup_motors, set_speeds, slew_toward, stop_motors
from indicators import (
    setup_indicators, new_led_state, start_blink, update_led,
    new_buzzer_state, play_tone, update_buzzer,
)
from utils import clamp
from debug_view import draw_debug_view, draw_roi_arrow_view
from buffer import TemporalBuffer
from adaptive_pid import schedule_pid_gains


def main():
    camera_kind, camera = open_camera()
    left_pwm, right_pwm = setup_motors()
    buzzer_pwm = setup_indicators()

    first_frame = read_frame(camera_kind, camera)
    if first_frame is not None:
        print_calibration_info(first_frame)

    integral = 0.0
    last_error = 0.0
    last_time = time.perf_counter()
    applied_forward = 0.0
    state = "FOLLOW"
    spin_start_time = None
    search_direction = 1.0
    mode = "AUTO"
    last_manual_key_time = time.perf_counter()
    last_manual_command = None

    red_led_state = new_led_state()
    green_led_state = new_led_state()
    buzzer_state = new_buzzer_state()
    prev_state = None
    prev_sharp_turn = False
    prev_red_marker = False
    prev_green_marker = False
    pending_marker_color = None
    pending_marker_action_time = None
    halted = False
    red_led_manual = False
    green_led_manual = False

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

    print("PID line following running. Press Ctrl+C to stop.")
    print("Manual indicator test keys: r=toggle red LED, g=toggle green LED, b=buzzer blip")
    print(f"Starting speed: BASE_SPEED={BASE_SPEED}  MAX_SPEED={MAX_SPEED}  MIN_SPEED={MIN_SPEED}")

    try:
        while True:
            now = time.perf_counter()
            dt = max(0.001, now - last_time)
            last_time = now
            max_step = SLEW_RATE_PER_S * dt
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
                    last_manual_command = None
                    pending_marker_color = None
                    pending_marker_action_time = None
                    backtrack_intersection_wp = None
                    backtrack_target_branch_idx = None
                    backtrack_start_time = None
                    rotate_settle_until = None
                    dead_end_recorded = False
                    stop_motors(left_pwm, right_pwm)
                elif key in (13, 10):
                    if mode != "AUTO":
                        print("SWITCHED TO AUTO: resuming line following")
                    mode = "AUTO"
                    state = "FOLLOW"
                    applied_forward = 0.0
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
                    stop_motors(left_pwm, right_pwm)
                    if halted:
                        print("MANUAL OVERRIDE: clearing red-marker halt")
                    halted = False
                elif mode == "MANUAL" and key in (ord("w"), ord("a"), ord("s"), ord("d")):
                    last_manual_key_time = now
                    last_manual_command = key
                elif key == ord("r"):
                    red_led_state["active"] = False
                    red_led_manual = not red_led_manual
                    GPIO.output(RED_LED_PIN, GPIO.HIGH if red_led_manual else GPIO.LOW)
                    print(f"RED LED manual: {'ON' if red_led_manual else 'OFF'}")
                elif key == ord("g"):
                    green_led_state["active"] = False
                    green_led_manual = not green_led_manual
                    GPIO.output(GREEN_LED_PIN, GPIO.HIGH if green_led_manual else GPIO.LOW)
                    print(f"GREEN LED manual: {'ON' if green_led_manual else 'OFF'}")
                elif key == ord("b"):
                    play_tone(buzzer_state, "manual_test")
                    print("BUZZER manual test tone")

            frame = read_frame(camera_kind, camera)
            if frame is None:
                continue

            if halted:
                stop_motors(left_pwm, right_pwm)
                update_led(red_led_state, RED_LED_PIN, now)
                update_led(green_led_state, GREEN_LED_PIN, now)
                update_buzzer(buzzer_state, buzzer_pwm, now)
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
                turn_component = clamp(target_turn, -MAX_TURN_SPEED, MAX_TURN_SPEED)
                applied_left = clamp(applied_forward + turn_component, -1.0, 1.0)
                applied_right = clamp(applied_forward - turn_component, -1.0, 1.0)
                set_speeds(applied_left, applied_right, left_pwm, right_pwm)

                update_led(red_led_state, RED_LED_PIN, now)
                update_led(green_led_state, GREEN_LED_PIN, now)
                update_buzzer(buzzer_state, buzzer_pwm, now)
                prev_state = state

                _, active_debug = find_line_error_normal(frame)
                if SHOW_DEBUG_VIEW and frame_count % DEBUG_VIEW_EVERY_N_FRAMES == 0:
                    draw_debug_view(frame, active_debug, None, None, applied_left, applied_right, state)
                    draw_roi_arrow_view(active_debug, applied_left, applied_right, state)

                if now - last_log_time >= LOOP_LOG_INTERVAL_S:
                    last_log_time = now
                    print(f"state={state:16s} applied L/R={applied_left:+.2f}/{applied_right:+.2f} fps={fps_ema:.1f}")
                continue

            # ===== AUTO MODE: VISION + STATE MACHINE =====
            normal_error, normal_debug = find_line_error_normal(frame)

            target_forward = 0.0
            target_turn = 0.0
            display_turn = None
            active_debug = normal_debug

            # INTERSECTION DETECTION (only when centered on the line)
            if (state not in ("BACKTRACK", "ROTATE")
                    and normal_error is not None
                    and abs(normal_error) < CENTER_DEADZONE * 2):
                if now - last_intersection_time > INTERSECTION_COOLDOWN_S:
                    intersection = detect_intersection_normal(frame)
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

            # ----- BACKTRACK: reverse along the line to reach last intersection -----
            if state == "BACKTRACK":
                backtrack_err, backtrack_debug = find_line_error_normal(frame)
                active_debug = backtrack_debug

                inter = detect_intersection_normal(frame)
                if inter:
                    print(f"BACKTRACK: reached intersection ({inter['branch_count']} branches)")
                    stop_motors(left_pwm, right_pwm)
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
                    wide_err, wide_debug = find_line_error_wide(frame)
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
                            stop_motors(left_pwm, right_pwm)
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
                        inter = detect_intersection_normal(frame)
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
                error = normal_error
                integral += error * dt
                integral = clamp(integral, -INTEGRAL_LIMIT, INTEGRAL_LIMIT)
                derivative = (error - last_error) / dt
                last_error = error

                line_confidence = normal_debug.get("line_confidence", 1.0)
                current_kp, current_ki, current_kd = schedule_pid_gains(
                    KP, KI, KD, error, derivative, line_confidence,
                )

                raw_turn = (current_kp * error) + (current_ki * integral) + (current_kd * derivative)
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

                derivative_load = clamp(
                    abs(derivative) / max(0.001, ADAPTIVE_DERIVATIVE_REF),
                    0.0, 1.0,
                )
                stability_scale = 1.0 - (ERROR_SPEED_REDUCTION * abs_error)
                stability_scale -= DERIVATIVE_SPEED_REDUCTION * derivative_load
                target_forward *= clamp(stability_scale, MIN_SPEED / max(BASE_SPEED, 0.001), 1.0)

                if line_confidence < 0.5:
                    confidence_scale = LOW_CONFIDENCE_SPEED_SCALE + (
                        (1.0 - LOW_CONFIDENCE_SPEED_SCALE) * line_confidence * 2.0
                    )
                    target_forward *= clamp(confidence_scale, LOW_CONFIDENCE_SPEED_SCALE, 1.0)

                target_forward = clamp(target_forward, MIN_SPEED, MAX_SPEED)

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

                wide_error, wide_debug = find_line_error_wide(frame)

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
                        target_turn = -SPIN_SEARCH_SPEED * search_direction
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

            # ----- MARKER ACTION (unchanged) -----
            if pending_marker_color is not None and now >= pending_marker_action_time:
                if pending_marker_color == "green":
                    print("GREEN MARKER: continuing")
                    start_blink(green_led_state, blinks=4, interval_s=0.12)
                    play_tone(buzzer_state, "marker_green")
                elif pending_marker_color == "red":
                    print("RED MARKER: stopping")
                    start_blink(red_led_state, blinks=None, interval_s=0.25, hold_on=True)
                    play_tone(buzzer_state, "marker_red")
                    stop_motors(left_pwm, right_pwm)
                    halted = True
                pending_marker_color = None
                pending_marker_action_time = None

            if halted:
                if SHOW_DEBUG_VIEW and frame_count % DEBUG_VIEW_EVERY_N_FRAMES == 0:
                    cv2.imshow("Camera + Decisions", frame)
                update_led(red_led_state, RED_LED_PIN, now)
                update_led(green_led_state, GREEN_LED_PIN, now)
                update_buzzer(buzzer_state, buzzer_pwm, now)
                continue

            # ----- STATE TRANSITION SOUNDS (unchanged) -----
            if state != prev_state and state in ("SPIN_SEARCH", "APPROACH", "FOLLOW"):
                play_tone(buzzer_state, {
                    "SPIN_SEARCH": "state_spin_search",
                    "APPROACH": "state_approach",
                    "FOLLOW": "state_follow",
                }[state])
            elif state != prev_state and state in ("BACKTRACK", "ROTATE"):
                play_tone(buzzer_state, "sharp_turn")

            is_sharp_turn = state == "FOLLOW" and abs(target_turn) > SHARP_TURN_SPEED * 0.6
            if is_sharp_turn and not prev_sharp_turn:
                play_tone(buzzer_state, "sharp_turn")
            prev_state = state
            prev_sharp_turn = is_sharp_turn

            # ----- APPLY SPEEDS -----
            if state == "BACKTRACK":
                applied_forward = slew_toward(applied_forward, target_forward, max_step)
                turn_component = clamp(target_turn, -MAX_TURN_SPEED, MAX_TURN_SPEED)
                applied_left = clamp(applied_forward + turn_component, -1.0, 1.0)
                applied_right = clamp(applied_forward - turn_component, -1.0, 1.0)
                set_speeds(applied_left, applied_right, left_pwm, right_pwm)

            elif state == "ROTATE":
                applied_forward = 0.0
                turn_component = clamp(target_turn, -ROTATE_SPEED, ROTATE_SPEED)
                applied_left = turn_component
                applied_right = -turn_component
                set_speeds(applied_left, applied_right, left_pwm, right_pwm)

            elif state == "STOP":
                applied_forward = 0.0
                turn_component = 0.0
                applied_left = 0.0
                applied_right = 0.0
                stop_motors(left_pwm, right_pwm)

            elif state == "SPIN_SEARCH":
                applied_forward = slew_toward(applied_forward, target_forward, max_step)
                turn_component = clamp(target_turn, -MAX_TURN_SPEED, MAX_TURN_SPEED)
                applied_left = clamp(applied_forward + turn_component, -1.0, 1.0)
                applied_right = clamp(applied_forward - turn_component, -1.0, 1.0)
                set_speeds(applied_left, applied_right, left_pwm, right_pwm)

            else:
                applied_forward = slew_toward(applied_forward, target_forward, max_step)
                turn_component = clamp(target_turn, -MAX_TURN_SPEED, MAX_TURN_SPEED)
                applied_left = clamp(applied_forward + turn_component, -1.0, 1.0)
                applied_right = clamp(applied_forward - turn_component, -1.0, 1.0)
                set_speeds(applied_left, applied_right, left_pwm, right_pwm)

            # ----- LED / BUZZER UPDATES -----
            update_led(red_led_state, RED_LED_PIN, now)
            update_led(green_led_state, GREEN_LED_PIN, now)
            update_buzzer(buzzer_state, buzzer_pwm, now)

            if SHOW_DEBUG_VIEW and frame_count % DEBUG_VIEW_EVERY_N_FRAMES == 0:
                draw_debug_view(
                    frame, active_debug,
                    normal_error if state == "FOLLOW" else None,
                    display_turn, applied_left, applied_right, state,
                )
                lean_ratio = (display_turn / TURN_LIMIT) if display_turn is not None else None
                draw_roi_arrow_view(active_debug, applied_left, applied_right, state, lean_ratio)

            if now - last_log_time >= LOOP_LOG_INTERVAL_S:
                last_log_time = now
                loop_ms = dt * 1000.0
                print(
                    f"state={state:12s} forward={applied_forward:+.2f} turn={turn_component:+.2f}  "
                    f"L/R={applied_left:+.2f}/{applied_right:+.2f}  loop_ms={loop_ms:.1f} "
                    f"fps={fps_ema:.1f} pid={current_kp:.2f}/{current_ki:.2f}/{current_kd:.2f}"
                )

    except KeyboardInterrupt:
        print("stopping")

    finally:
        stop_motors(left_pwm, right_pwm)
        left_pwm.stop()
        right_pwm.stop()
        GPIO.output(RED_LED_PIN, GPIO.LOW)
        GPIO.output(GREEN_LED_PIN, GPIO.LOW)
        buzzer_pwm.stop()
        GPIO.cleanup()
        if camera_kind == "picamera2":
            camera.stop()
        else:
            camera.release()
        if SHOW_DEBUG_VIEW:
            cv2.destroyAllWindows()
        print("cleaned up")


if __name__ == "__main__":
    main()
