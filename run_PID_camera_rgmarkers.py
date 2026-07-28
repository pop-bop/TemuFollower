#!/usr/bin/env python3
import math
import time

import cv2
import numpy as np

import RPi.GPIO as GPIO


#PID

# KP: how hard the robot steers based on how far off center the line is.
# Change this when: the robot drifts off curves or reacts too late -> raise KP.
# Change this when: the robot zigzags/wobbles even on straight lines -> lower KP.
KP = 1.10
# KI: corrects small steady drift over time.
# Change this when: the robot always leans to one side even on a straight 
# line, even though the line looks centered -> raise KI slightly (0.01 steps).
KI = 0.00
# KD: reacts to how fast the error is changing, smooths out steering.
# Change this when: the robot oscillates/wobbles side to side on straights
# -> raise KD.
# Change this when: the robot feels sluggish or slow to respond on curves
# -> lower KD.
KD = 0.18

TURN_LIMIT = 0.80

# it still turns too gently on real curves -> lower
# it turns too sharply on small, centered wobble -> raise CENTER_DEADZONE.
CENTER_DEADZONE = 0.12


#SPEEDS
SHARP_TURN_SPEED = 0.32
MAX_TURN_SPEED = 0.40
STEER_INVERT = True # this i added because the wheels were turning the opposite way

BASE_SPEED = 0.25
MAX_SPEED = 0.45
MIN_SPEED = 0.15

SPIN_SEARCH_SPEED = 0.26
APPROACH_SPEED = 0.22

# if still not found after spinning this many seconds, stop completely instead of spinning forever. 
LINE_LOST_STOP_TIMEOUT_S = 4.0

#the increase & decrease of speed
SLEW_RATE_PER_S = 0.3

MANUAL_SPEED = 0.22
MANUAL_KEY_TIMEOUT_S = 0.5


#ROI VALUES



ROI_Y_START_RATIO = 0.43
ROI_Y_END_RATIO   = 0.86
ROI_X_START_RATIO = 0.04
ROI_X_END_RATIO   = 0.98

WIDE_ROI_Y_START_RATIO = 0.20
WIDE_ROI_X_START_RATIO = 0.0
WIDE_ROI_X_END_RATIO = 1.0


#THRESHOLDS
BLACK_THRESHOLD = 75

#smallest blob size counted as a real line, filters out noise.
MIN_LINE_AREA = 45

# If the debug camera view shows a real red marker as blue, flip this to
# False. Green will still look roughly green when red/blue are swapped,
# which is why this can be confusing at first.
PICAMERA2_RGB_TO_BGR = False

# Marker detection uses BGR channel dominance instead of HSV hue. This is
# much easier to tune when the camera/lighting shifts the exact shade:
# green = G is stronger than R and B; red = R is stronger than G and B.
GREEN_DIFF_THRESHOLD = 40
RED_DIFF_THRESHOLD = 40


# Camera settings
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 120

# Motor pins, same as motor_test.py
LEFT_ENA = 12
LEFT_IN1 = 16
LEFT_IN2 = 20
RIGHT_ENB = 18
RIGHT_IN3 = 21
RIGHT_IN4 = 26

PWM_FREQUENCY_HZ = 1000

# Indicator pins (not wired yet -- pick free BCM pins and wire accordingly).
RED_LED_PIN = 5
GREEN_LED_PIN = 6
BUZZER_PIN = 13  # passive buzzer assumed (PWM tone); see setup_indicators()

# smallest red/green blob touching the line counted as a real marker.
MIN_MARKER_AREA = 40

# Delay between first seeing a red/green marker and firing its LED/buzzer
# action. This lets the robot drive until the marker is physically under it.
MARKER_ACTION_DELAY_S = 1.0





def clamp(value, low, high):
    return max(low, min(high, value))


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
        if PICAMERA2_RGB_TO_BGR:
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return rgb
    else:
        ok, frame = camera.read()
        return frame if ok else None


def find_line_error(frame, y_start_ratio, y_end_ratio, x_start_ratio, x_end_ratio):
    h, w = frame.shape[:2]
    y0 = int(h * y_start_ratio)
    y1 = int(h * y_end_ratio)
    x0 = int(w * x_start_ratio)
    x1 = int(w * x_end_ratio)
    roi = frame[y0:y1, x0:x1]
    roi_w = x1 - x0

    b, g, r = cv2.split(roi)
    _, gr = cv2.threshold(cv2.subtract(g, r), GREEN_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    _, gb = cv2.threshold(cv2.subtract(g, b), GREEN_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    green_mask = cv2.bitwise_and(gr, gb)
    _, rg = cv2.threshold(cv2.subtract(r, g), RED_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    _, rb = cv2.threshold(cv2.subtract(r, b), RED_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    red_mask = cv2.bitwise_and(rg, rb)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, black_mask = cv2.threshold(blur, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    # Exclude red and green marker pixels so they are never mistaken for
    # the black line.
    black_mask[green_mask > 0] = 0
    black_mask[red_mask > 0] = 0

    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    debug_info = {
        "roi_bounds": (x0, y0, x1, y1),
        "mask": black_mask,
        "line_point": None,
        "roi_frame": roi,
        "red_marker": False,
        "green_marker": False,
    }

    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, debug_info

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_LINE_AREA:
        return None, debug_info

    # Check whether a red/green marker is touching (or overlapping) the
    # line itself, not just visible somewhere in the ROI: dilate the
    # line contour into a "near-line" zone and intersect it with the
    # marker masks computed above.
    near_line_zone = np.zeros(black_mask.shape, dtype=np.uint8)
    cv2.drawContours(near_line_zone, [largest], -1, 255, thickness=cv2.FILLED)
    near_line_zone = cv2.dilate(near_line_zone, np.ones((15, 15), np.uint8))

    red_hit = cv2.bitwise_and(near_line_zone, red_mask)
    green_hit = cv2.bitwise_and(near_line_zone, green_mask)
    debug_info["red_marker"] = cv2.countNonZero(red_hit) >= MIN_MARKER_AREA
    debug_info["green_marker"] = cv2.countNonZero(green_hit) >= MIN_MARKER_AREA

    M = cv2.moments(largest)
    if M["m00"] <= 0:
        return None, debug_info

    cx_roi = int(M["m10"] / M["m00"])
    cy_roi = int(M["m01"] / M["m00"])
    cx_global = cx_roi + x0
    cy_global = cy_roi + y0
    center_x_global = x0 + roi_w // 2
    error = (cx_global - center_x_global) / max(1.0, roi_w / 2.0)
    debug_info["line_point"] = (cx_global, cy_global)
    return clamp(error, -1.0, 1.0), debug_info


def find_line_error_normal(frame):
    return find_line_error(frame, ROI_Y_START_RATIO, ROI_Y_END_RATIO, ROI_X_START_RATIO, ROI_X_END_RATIO)


def find_line_error_wide(frame):
    # Wide search still goes all the way to the bottom of the frame
    # (y_end_ratio=1.0), since it is meant to look further out, not a
    # small bounded box like the normal ROI.
    return find_line_error(frame, WIDE_ROI_Y_START_RATIO, 1.0, WIDE_ROI_X_START_RATIO, WIDE_ROI_X_END_RATIO)


# MOTORS
def setup_motors():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    GPIO.setup(LEFT_IN1, GPIO.OUT)
    GPIO.setup(LEFT_IN2, GPIO.OUT)
    GPIO.setup(RIGHT_IN3, GPIO.OUT)
    GPIO.setup(RIGHT_IN4, GPIO.OUT)
    GPIO.setup(LEFT_ENA, GPIO.OUT)
    GPIO.setup(RIGHT_ENB, GPIO.OUT)

    left_pwm = GPIO.PWM(LEFT_ENA, PWM_FREQUENCY_HZ)
    right_pwm = GPIO.PWM(RIGHT_ENB, PWM_FREQUENCY_HZ)
    left_pwm.start(0)
    right_pwm.start(0)
    return left_pwm, right_pwm


def set_motor_speed(speed, in_a, in_b, pwm, invert=False):
    forward = speed >= 0
    if invert:
        forward = not forward
    if forward:
        GPIO.output(in_a, GPIO.HIGH)
        GPIO.output(in_b, GPIO.LOW)
    else:
        GPIO.output(in_a, GPIO.LOW)
        GPIO.output(in_b, GPIO.HIGH)
    pwm.ChangeDutyCycle(clamp(abs(speed), 0.0, 1.0) * 100.0)


def set_speeds(left, right, left_pwm, right_pwm):
    left = clamp(left, -1.0, 1.0)
    right = clamp(right, -1.0, 1.0)
    set_motor_speed(left, LEFT_IN1, LEFT_IN2, left_pwm)
    # Right motor wiring is inverted compared to the left motor: forward
    # is IN3=LOW, IN4=HIGH (opposite of the left motor's pattern), so
    # invert=True flips the polarity here.
    set_motor_speed(right, RIGHT_IN3, RIGHT_IN4, right_pwm, invert=True)


def slew_toward(current, target, max_step):
    # Move current a small step toward target, capped by max_step, so the
    # applied speed ramps smoothly instead of jumping straight to target.
    if target > current:
        return min(target, current + max_step)
    if target < current:
        return max(target, current - max_step)
    return current


def stop_motors(left_pwm, right_pwm):
    set_speeds(0.0, 0.0, left_pwm, right_pwm)


# INDICATORS (LEDs + buzzer)
def setup_indicators():
    GPIO.setup(RED_LED_PIN, GPIO.OUT)
    GPIO.setup(GREEN_LED_PIN, GPIO.OUT)
    GPIO.setup(BUZZER_PIN, GPIO.OUT)
    GPIO.output(RED_LED_PIN, GPIO.LOW)
    GPIO.output(GREEN_LED_PIN, GPIO.LOW)
    # Passive buzzer assumed: PWM lets us vary pitch per event. If the
    # buzzer turns out to be active (on/off only, fixed pitch), replace
    # the ChangeFrequency/ChangeDutyCycle calls in update_buzzer() with
    # plain GPIO.output(BUZZER_PIN, HIGH/LOW) -- the on/off timing in
    # TONES and the sequencing logic below don't need to change.
    buzzer_pwm = GPIO.PWM(BUZZER_PIN, 440)
    buzzer_pwm.start(0)
    return buzzer_pwm


def new_led_state():
    return {"active": False, "on": False, "next_toggle": 0.0,
            "interval": 0.1, "blinks_left": 0, "hold_on": False}


def start_blink(led_state, blinks, interval_s, hold_on=False):
    # blinks=None means blink forever (used for the red halt indicator).
    led_state.update(active=True, on=False, next_toggle=0.0,
                      interval=interval_s, blinks_left=blinks, hold_on=hold_on)


def update_led(led_state, pin, now):
    if not led_state["active"] or now < led_state["next_toggle"]:
        return
    led_state["on"] = not led_state["on"]
    GPIO.output(pin, GPIO.HIGH if led_state["on"] else GPIO.LOW)
    led_state["next_toggle"] = now + led_state["interval"]
    if led_state["blinks_left"] is not None:
        led_state["blinks_left"] -= 1
        if led_state["blinks_left"] <= 0:
            led_state["active"] = False
            GPIO.output(pin, GPIO.HIGH if led_state["hold_on"] else GPIO.LOW)


# name: list of (freq_hz, on_ms, off_ms) segments played in order, used
# as a little audible "debug panel" -- each notable event/state gets
# its own distinct sound.
TONES = {
    "state_follow":      [(880, 40, 0)],
    "state_approach":    [(600, 70, 40), (600, 70, 0)],
    "state_spin_search": [(300, 120, 60), (300, 120, 60), (300, 120, 0)],
    "sharp_turn":        [(1000, 35, 0)],
    "marker_green":      [(784, 90, 40), (988, 120, 0)],
    "marker_red":        [(500, 180, 80), (320, 260, 0)],
    "manual_test":       [(660, 100, 0)],
}


def new_buzzer_state():
    return {"queue": [], "seg_start": None}


def play_tone(buzzer_state, name):
    buzzer_state["queue"] = list(TONES[name])
    buzzer_state["seg_start"] = None


def update_buzzer(buzzer_state, buzzer_pwm, now):
    if not buzzer_state["queue"]:
        return
    freq, on_ms, off_ms = buzzer_state["queue"][0]
    if buzzer_state["seg_start"] is None:
        buzzer_state["seg_start"] = now
        buzzer_pwm.ChangeFrequency(freq)
        buzzer_pwm.ChangeDutyCycle(50)
    elapsed_ms = (now - buzzer_state["seg_start"]) * 1000.0
    if elapsed_ms < on_ms:
        return
    if elapsed_ms < on_ms + off_ms:
        buzzer_pwm.ChangeDutyCycle(0)
        return
    buzzer_state["queue"].pop(0)
    buzzer_state["seg_start"] = None
    buzzer_pwm.ChangeDutyCycle(0)


# CALIBRATION CHECK
def print_calibration_info(frame):
    h, w = frame.shape[:2]
    x0 = int(w * ROI_X_START_RATIO)
    x1 = int(w * ROI_X_END_RATIO)
    center_x_global = x0 + (x1 - x0) // 2
    print("Calibration check:")
    print(f"  frame size: {w}x{h}")
    print(f"  ROI x range: {x0} to {x1}")
    print(f"  ROI center x (should match robot centerline): {center_x_global}")
    print(f"  frame center x: {w // 2}")
    if abs(center_x_global - w // 2) > 2:
        print("  NOTE: ROI center is not the same as frame center. This is expected")
        print("  only if the camera is intentionally offset. Otherwise consider")
        print("  adjusting ROI_X_START_RATIO / ROI_X_END_RATIO or the camera mount.")



# DEBUG_VIEW

SHOW_DEBUG_VIEW = True
ROI_VIEW_SCALE = 3
ARROW_MAX_DEFLECTION_DEG = 65.0


def draw_debug_view(frame, debug_info, error, turn, left, right, state):
    x0, y0, x1, y1 = debug_info["roi_bounds"]
    debug = frame.copy()

    cv2.rectangle(debug, (x0, y0), (x1, y1), (0, 255, 255), 1)
    center_x = x0 + (x1 - x0) // 2
    cv2.line(debug, (center_x, y0), (center_x, y1), (255, 0, 0), 1)

    if debug_info["line_point"] is not None:
        cv2.circle(debug, debug_info["line_point"], 5, (0, 0, 255), -1)

    lines = [
        f"state: {state}",
        f"error: {error:+.3f}" if error is not None else "error: none",
        f"turn:  {turn:+.3f}" if turn is not None else "turn:  none",
        f"L/R:   {left:+.2f} / {right:+.2f}" if left is not None else "L/R:   none",
    ]
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
        # Positive diff (left faster than right) means the robot is
        # turning right, matching the turn convention used in the main
        # loop (turn>0 increases left speed and decreases right speed).
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


# MAIN LOOP
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

    print("PID line following running. Press Ctrl+C to stop.")
    print("Manual indicator test keys: r=toggle red LED, g=toggle green LED, b=buzzer blip")
    print(f"Starting speed: BASE_SPEED={BASE_SPEED}  MAX_SPEED={MAX_SPEED}  MIN_SPEED={MIN_SPEED}")

    try:
        while True:
            now = time.perf_counter()
            dt = max(0.001, now - last_time)
            last_time = now
            max_step = SLEW_RATE_PER_S * dt

            key = -1
            if SHOW_DEBUG_VIEW:
                # SPACE = stop immediately and switch to MANUAL (WASD).
                # ENTER = stop immediately and switch back to AUTO
                # (normal line following, also clears a red-marker halt).
                # r/g = toggle red/green LED on/off manually (for wiring
                # tests). b = play a short buzzer test blip. All of this
                # only works while a debug window has keyboard focus.
                key = cv2.waitKey(1) & 0xFF
                if key == 32:
                    if mode != "MANUAL":
                        print("SWITCHED TO MANUAL: use w/a/s/d, space=stop, enter=back to auto")
                    mode = "MANUAL"
                    applied_forward = 0.0
                    last_manual_command = None
                    pending_marker_color = None
                    pending_marker_action_time = None
                    stop_motors(left_pwm, right_pwm)
                elif key in (13, 10):
                    if mode != "AUTO":
                        print("SWITCHED TO AUTO: resuming line following")
                    mode = "AUTO"
                    applied_forward = 0.0
                    pending_marker_color = None
                    pending_marker_action_time = None
                    stop_motors(left_pwm, right_pwm)
                    if halted:
                        print("MANUAL OVERRIDE: clearing red-marker halt")
                    halted = False
                elif mode == "MANUAL" and key in (ord("w"), ord("a"), ord("s"), ord("d")):
                    last_manual_key_time = now
                    last_manual_command = key
                elif key == ord("r"):
                    # Manual test/override for the red LED: toggle it on
                    # solid or off, independent of the marker/halt logic.
                    red_led_state["active"] = False
                    red_led_manual = not red_led_manual
                    GPIO.output(RED_LED_PIN, GPIO.HIGH if red_led_manual else GPIO.LOW)
                    print(f"RED LED manual: {'ON' if red_led_manual else 'OFF'}")
                elif key == ord("g"):
                    # Manual test/override for the green LED.
                    green_led_state["active"] = False
                    green_led_manual = not green_led_manual
                    GPIO.output(GREEN_LED_PIN, GPIO.HIGH if green_led_manual else GPIO.LOW)
                    print(f"GREEN LED manual: {'ON' if green_led_manual else 'OFF'}")
                elif key == ord("b"):
                    # Manual test blip on the buzzer.
                    play_tone(buzzer_state, "manual_test")
                    print("BUZZER manual test tone")

            frame = read_frame(camera_kind, camera)
            if frame is None:
                continue

            if halted:
                # Red marker was reached: stay stopped, keep servicing
                # the LED/buzzer sequencers, but skip PID/vision.
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
                if SHOW_DEBUG_VIEW:
                    draw_debug_view(frame, active_debug, None, None, applied_left, applied_right, state)
                    draw_roi_arrow_view(active_debug, applied_left, applied_right, state)

                print(f"state={state:16s} applied L/R={applied_left:+.2f}/{applied_right:+.2f}")
                continue

            normal_error, normal_debug = find_line_error_normal(frame)

            target_forward = 0.0
            target_turn = 0.0
            display_turn = None  # pre-invert turn, for the arrow only
            active_debug = normal_debug

            if normal_error is not None:
                # Line visible in the normal, close-in ROI. Resume/stay
                # in FOLLOW no matter what state we were in before.
                state = "FOLLOW"
                error = normal_error
                integral += error * dt
                derivative = (error - last_error) / dt
                last_error = error

                raw_turn = (KP * error) + (KI * integral) + (KD * derivative)
                raw_turn = clamp(raw_turn, -TURN_LIMIT, TURN_LIMIT)
                display_turn = raw_turn
                motor_turn = -raw_turn if STEER_INVERT else raw_turn

                abs_error = abs(error)
                if abs_error <= CENTER_DEADZONE:
                    # Smooth steering: forward stays at BASE_SPEED, the
                    # turn component blends both wheels around it.
                    target_forward = BASE_SPEED
                    target_turn = motor_turn
                else:
                    # Beyond the deadzone, blend progressively toward
                    # true opposite-wheel turning (forward component
                    # shrinks toward 0, turn component grows toward
                    # SHARP_TURN_SPEED), since smooth blended steering
                    # alone was too gentle to actually follow real curves.
                    blend = clamp(
                        (abs_error - CENTER_DEADZONE) / max(0.001, 1.0 - CENTER_DEADZONE),
                        0.0, 1.0,
                    )
                    turn_sign = 1.0 if motor_turn >= 0 else -1.0
                    target_forward = BASE_SPEED * (1.0 - blend)
                    target_turn = motor_turn * (1.0 - blend) + (SHARP_TURN_SPEED * turn_sign) * blend

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

            else:
                # Normal ROI lost the line: no marker can be "on the
                # line" right now, so clear the debounce state.
                prev_green_marker = False
                prev_red_marker = False

                # Normal ROI lost the line. Check the wide ROI to see if
                # it is visible further ahead.
                wide_error, wide_debug = find_line_error_wide(frame)

                if wide_error is not None:
                    # Found further away, creep forward gently toward it.
                    state = "APPROACH"
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
                    # Not visible in either ROI, spin in place to search.
                    state = "SPIN_SEARCH"
                    active_debug = wide_debug
                    if spin_start_time is None:
                        spin_start_time = now

                    if now - spin_start_time < LINE_LOST_STOP_TIMEOUT_S:
                        target_forward = 0.0
                        target_turn = -SPIN_SEARCH_SPEED * search_direction
                    else:
                        # Give up spinning, sit still, but keep checking
                        # every loop in case the line reappears.
                        target_forward = 0.0
                        target_turn = 0.0

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
                # A red marker fired above: stay stopped this iteration
                # too instead of letting the speed-application code
                # below drive off again.
                if SHOW_DEBUG_VIEW:
                    cv2.imshow("Camera + Decisions", frame)
                update_led(red_led_state, RED_LED_PIN, now)
                update_led(green_led_state, GREEN_LED_PIN, now)
                update_buzzer(buzzer_state, buzzer_pwm, now)
                continue

            # Audible "debug panel": a distinct tone the moment the
            # state machine switches into FOLLOW/APPROACH/SPIN_SEARCH,
            # and a short blip on the rising edge of a sharp turn.
            if state != prev_state and state in ("SPIN_SEARCH", "APPROACH", "FOLLOW"):
                play_tone(buzzer_state, {
                    "SPIN_SEARCH": "state_spin_search",
                    "APPROACH": "state_approach",
                    "FOLLOW": "state_follow",
                }[state])

            is_sharp_turn = state == "FOLLOW" and abs(target_turn) > SHARP_TURN_SPEED * 0.6
            if is_sharp_turn and not prev_sharp_turn:
                play_tone(buzzer_state, "sharp_turn")
            prev_state = state
            prev_sharp_turn = is_sharp_turn

            # Only the forward component is slew-rate smoothed. The turn
            # component is applied instantly every loop, no ramp delay,
            # so steering reacts the moment the camera sees an error,
            # regardless of how smooth/slow the forward speed ramp is.
            if state == "SPIN_SEARCH":
                applied_forward = slew_toward(applied_forward, target_forward, max_step)
                turn_component = clamp(target_turn, -MAX_TURN_SPEED, MAX_TURN_SPEED)
                applied_left = clamp(applied_forward + turn_component, -1.0, 1.0)
                applied_right = clamp(applied_forward - turn_component, -1.0, 1.0)
                set_speeds(-0.2, 0.2, left_pwm, right_pwm)
            else:
                applied_forward = slew_toward(applied_forward, target_forward, max_step)
                turn_component = clamp(target_turn, -MAX_TURN_SPEED, MAX_TURN_SPEED)
                applied_left = clamp(applied_forward + turn_component, -1.0, 1.0)
                applied_right = clamp(applied_forward - turn_component, -1.0, 1.0)
                set_speeds(applied_left, applied_right, left_pwm, right_pwm)

            update_led(red_led_state, RED_LED_PIN, now)
            update_led(green_led_state, GREEN_LED_PIN, now)
            update_buzzer(buzzer_state, buzzer_pwm, now)

            if SHOW_DEBUG_VIEW:
                draw_debug_view(
                    frame, active_debug,
                    normal_error if state == "FOLLOW" else None,
                    display_turn, applied_left, applied_right, state,
                )
                lean_ratio = (display_turn / TURN_LIMIT) if display_turn is not None else None
                draw_roi_arrow_view(active_debug, applied_left, applied_right, state, lean_ratio)

            loop_ms = dt * 1000.0
            print(
                f"state={state:12s} forward={applied_forward:+.2f} turn={turn_component:+.2f}  "
                f"applied L/R={applied_left:+.2f}/{applied_right:+.2f}  loop_ms={loop_ms:.1f}"
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
