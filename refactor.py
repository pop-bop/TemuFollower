import re

with open("main.py", "r") as f:
    content = f.read()

# 1. Imports
content = re.sub(
    r"from motors import setup_motors, set_speeds, slew_toward, stop_motors\nfrom indicators import \(\n    setup_indicators, new_led_state, start_blink, update_led,\n    new_buzzer_state, play_tone, update_buzzer,\n\)",
    "from spi import SPIController, pack_movement, pack_interrupt, pack_speciality, pack_grabber\nfrom vision import get_warp_matrix, warp_frame, get_expected_depth_map, get_obstacle_mask\nfrom camera import get_intrinsics, get_depth_scale\nfrom config import CAMERA_TILT_ANGLE_DEG, CAMERA_MOUNT_HEIGHT_MM, SPI_BUS, SPI_DEVICE, SPI_MAX_SPEED_HZ\n\ndef slew_toward(current, target, max_step):\n    if target > current: return min(target, current + max_step)\n    if target < current: return max(target, current - max_step)\n    return current",
    content
)

# 2. Setup
content = re.sub(
    r"    camera_kind, camera = open_camera\(\)\n    left_pwm, right_pwm = setup_motors\(\)\n    buzzer_pwm = setup_indicators\(\)\n\n    first_frame = read_frame\(camera_kind, camera\)\n    if first_frame is not None:\n        print_calibration_info\(first_frame\)",
    "    camera_kind, camera = open_camera()\n    depth_scale = get_depth_scale(camera)\n    spi = SPIController(bus=SPI_BUS, device=SPI_DEVICE, max_speed_hz=SPI_MAX_SPEED_HZ)\n    warp_M = get_warp_matrix()\n    expected_depth_map = None\n\n    first_color, first_depth = read_frame(camera_kind, camera)\n    if first_color is not None:\n        h, w = first_color.shape[:2]\n        expected_depth_map = get_expected_depth_map(w, h, CAMERA_TILT_ANGLE_DEG, CAMERA_MOUNT_HEIGHT_MM)\n        warped = warp_frame(first_color, warp_M)\n        print_calibration_info(warped)",
    content
)

# 3. Read frame loop
content = re.sub(
    r"            frame = read_frame\(camera_kind, camera\)\n            if frame is None:\n                continue\n\n            if halted:\n                stop_motors\(left_pwm, right_pwm\)\n                prev_speed_for_accel = 0.0\n                update_led\(red_led_state, RED_LED_PIN, now\)\n                update_led\(green_led_state, GREEN_LED_PIN, now\)\n                update_buzzer\(buzzer_state, buzzer_pwm, now, False\)\n                if SHOW_DEBUG_VIEW:\n                    cv2.imshow\(\"Camera \+ Decisions\", frame\)\n                continue",
    "            color_frame, depth_frame = read_frame(camera_kind, camera)\n            if color_frame is None:\n                continue\n\n            frame = warp_frame(color_frame, warp_M)\n            if expected_depth_map is not None and depth_frame is not None:\n                warped_depth = warp_frame(depth_frame, warp_M)\n                obstacle_mask = get_obstacle_mask(warped_depth, depth_scale, expected_depth_map)\n            else:\n                obstacle_mask = None\n\n            if halted:\n                spi.transmit(pack_interrupt())\n                prev_speed_for_accel = 0.0\n                if SHOW_DEBUG_VIEW:\n                    cv2.imshow(\"Camera + Decisions\", frame)\n                continue",
    content
)

# 4. Stop motors calls & mode switches
content = re.sub(r"stop_motors\(left_pwm, right_pwm\)", "spi.transmit(pack_interrupt())", content)

# 5. find_line_error_normal(frame) -> find_line_error_normal(frame, obstacle_mask)
content = re.sub(r"find_line_error_normal\(frame\)", "find_line_error_normal(frame, obstacle_mask)", content)
content = re.sub(r"find_line_error_lookahead\(frame\)", "find_line_error_lookahead(frame, obstacle_mask)", content)
content = re.sub(r"find_line_error_far\(frame\)", "find_line_error_far(frame, obstacle_mask)", content)
content = re.sub(r"find_line_error_wide\(frame\)", "find_line_error_wide(frame, obstacle_mask)", content)
content = re.sub(r"detect_intersection_normal\(frame\)", "detect_intersection_normal(frame, obstacle_mask)", content)

# 6. Apply Speeds logic (end of loop)
# We want to replace the whole APPLY SPEEDS block with generating RK4 waypoints and sending SPI.
# And remove the update_led / update_buzzer block.
# Let's just find the apply speeds block and replace it.
apply_speeds_start = content.find("            # ----- APPLY SPEEDS -----")
led_buzzer_end = content.find("            if SHOW_DEBUG_VIEW and frame_count % DEBUG_VIEW_EVERY_N_FRAMES == 0:")

if apply_speeds_start != -1 and led_buzzer_end != -1:
    new_apply = """            # ----- SPI WAYPOINT GENERATION & TRANSMIT -----
            from trajectory import generate_waypoints
            
            if state == "STOP":
                spi.transmit(pack_interrupt())
                applied_forward = 0.0
                applied_left = 0.0
                applied_right = 0.0
                display_turn = 0.0
            else:
                applied_forward = slew_toward(applied_forward, target_forward, max_step)
                
                # Generate 5 waypoints looking ahead 1.0s (0.2s each)
                # Curve and target_error should be updated.
                waypoints = generate_waypoints(
                    trajectory_state, 
                    normal_error if normal_error is not None else 0.0,
                    curve_sharpness if curve_sharpness is not None else 0.0,
                    applied_forward, 
                    dt=0.2, num=5, kp=current_kp, steer_invert=STEER_INVERT
                )
                spi.transmit(pack_movement(waypoints))
                
                # Update visual variables for debug UI
                turn_component = clamp(target_turn, -MAX_TURN_SPEED, MAX_TURN_SPEED)
                applied_left = clamp(applied_forward + turn_component, -1.0, 1.0)
                applied_right = clamp(applied_forward - turn_component, -1.0, 1.0)

            # Speciality packets for LEDs/Buzzer events
            if pending_marker_color is not None and now >= pending_marker_action_time:
                pass # Handled above in marker action section now, wait... I should fix that section.

"""
    content = content[:apply_speeds_start] + new_apply + content[led_buzzer_end:]

# Fix marker action section
marker_action = """            if pending_marker_color is not None and now >= pending_marker_action_time:
                if pending_marker_color == "green":
                    print("GREEN MARKER: continuing")
                    spi.transmit(pack_speciality(0)) # 00
                elif pending_marker_color == "red":
                    print("RED MARKER: stopping")
                    spi.transmit(pack_speciality(1)) # 01
                    spi.transmit(pack_interrupt())
                    halted = True
                pending_marker_color = None
                pending_marker_action_time = None
"""
content = re.sub(r"            # ----- MARKER ACTION \(unchanged\) -----.*?pending_marker_action_time = None", marker_action, content, flags=re.DOTALL)

# Fix sound events
sounds = """            # ----- STATE TRANSITION SOUNDS -----
            if state != prev_state and state in ("SPIN_SEARCH", "APPROACH", "FOLLOW", "BACKTRACK", "ROTATE"):
                spi.transmit(pack_speciality(2)) # 10 = buzzer
"""
content = re.sub(r"            # ----- STATE TRANSITION SOUNDS \(unchanged\) -----.*?prev_sharp_turn = is_sharp_turn", sounds, content, flags=re.DOTALL)

# Fix cleanup
cleanup = """    finally:
        spi.transmit(pack_interrupt())
        if camera_kind == "picamera2":
            camera.stop()
        elif camera_kind == "realsense":
            camera[0].stop()
        else:
            if hasattr(camera, 'release'): camera.release()
        if SHOW_DEBUG_VIEW:
            cv2.destroyAllWindows()
        print("cleaned up")
"""
content = re.sub(r"    finally:.*?print\(\"cleaned up\"\)", cleanup, content, flags=re.DOTALL)

with open("main_refactored.py", "w") as f:
    f.write(content)
