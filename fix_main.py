with open('main.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    # Chunk 1
    if i >= 72 and i <= 84:
        if i == 72:
            new_lines.extend([
                '    prev_state = None\n',
                '    prev_sharp_turn = False\n',
                '    prev_red_marker = False\n',
                '    prev_green_marker = False\n',
                '    pending_marker_color = None\n',
                '    pending_marker_action_time = None\n',
                '    halted = False\n'
            ])
        continue
    # Chunk 2
    if i >= 171 and i <= 183:
        if i == 171:
            new_lines.extend([
                '                elif key == ord("r"):\n',
                '                    spi.transmit(pack_speciality(1))\n',
                '                    print("RED LED SPI trigger (0.5s)")\n',
                '                elif key == ord("g"):\n',
                '                    spi.transmit(pack_speciality(0))\n',
                '                    print("GREEN LED SPI trigger (0.5s)")\n',
                '                elif key == ord("b"):\n',
                '                    spi.transmit(pack_speciality(2))\n',
                '                    print("BUZZER manual test tone")\n'
            ])
        continue
    # Chunk 3
    if i >= 225 and i <= 238:
        if i == 225:
            new_lines.extend([
                '                applied_forward = slew_toward(applied_forward, target_forward, max_step)\n',
                '                turn_component = clamp(target_turn, -MAX_TURN_SPEED, MAX_TURN_SPEED)\n',
                '                applied_left = clamp(applied_forward + turn_component, -1.0, 1.0)\n',
                '                applied_right = clamp(applied_forward - turn_component, -1.0, 1.0)\n',
                '                \n',
                '                turn_int = int(turn_component * 127.0)\n',
                '                fwd_int = int(applied_forward * 127.0)\n',
                '                spi.transmit(pack_movement([(turn_int, fwd_int)] * 5))\n',
                '\n',
                '                current_speed_mag = abs(applied_forward)\n',
                '                accelerating = ACCEL_BUZZER_ENABLED and current_speed_mag > prev_speed_for_accel + ACCEL_SPEED_DELTA_THRESHOLD\n',
                '                prev_speed_for_accel = current_speed_mag\n',
                '\n',
                '                if accelerating:\n',
                '                    spi.transmit(pack_speciality(2))\n',
                '                prev_state = state\n'
            ])
        continue
    # Chunk 4
    if i >= 590 and i <= 597:
        if i == 590:
            new_lines.extend([
                '            if halted:\n',
                '                if SHOW_DEBUG_VIEW and frame_count % DEBUG_VIEW_EVERY_N_FRAMES == 0:\n',
                '                    cv2.imshow("Camera + Decisions", frame)\n',
                '                prev_speed_for_accel = 0.0\n',
                '                continue\n'
            ])
        continue
    # Chunk 5
    if i >= 632 and i <= 1079:
        if i == 632:
            new_lines.extend([
                '            if SHOW_DEBUG_VIEW and frame_count % DEBUG_VIEW_EVERY_N_FRAMES == 0:\n',
                '                extra_lines = None\n',
                '                lookahead_bounds = None\n',
                '                lookahead_point = None\n',
                '                far_bounds = None\n',
                '                far_point = None\n',
                '                if state == "FOLLOW" and lookahead_debug is not None:\n',
                '                    extra_lines = [f"curve={curve_sharpness:.2f} spd_scale={curve_speed_scale:.2f}"]\n',
                '                    lookahead_bounds = lookahead_debug.get("roi_bounds")\n',
                '                    lookahead_point = lookahead_debug.get("line_point")\n',
                '                    if far_debug is not None:\n',
                '                        far_bounds = far_debug.get("roi_bounds")\n',
                '                        far_point = far_debug.get("line_point")\n',
                '                draw_debug_view(\n',
                '                    frame, active_debug,\n',
                '                    normal_error if state == "FOLLOW" else None,\n',
                '                    display_turn, applied_left, applied_right, state,\n',
                '                    extra_lines=extra_lines,\n',
                '                    lookahead_bounds=lookahead_bounds,\n',
                '                    lookahead_point=lookahead_point,\n',
                '                    far_bounds=far_bounds,\n',
                '                    far_point=far_point,\n',
                '                )\n',
                '                lean_ratio = (display_turn / TURN_LIMIT) if display_turn is not None else None\n',
                '                draw_roi_arrow_view(active_debug, applied_left, applied_right, state, lean_ratio)\n',
                '\n',
                '            if now - last_log_time >= LOOP_LOG_INTERVAL_S:\n',
                '                last_log_time = now\n',
                '                loop_ms = dt * 1000.0\n',
                '                print(\n',
                '                    f"state={state:12s} forward={applied_forward:+.2f} turn={turn_component:+.2f}  "\n',
                '                    f"L/R={applied_left:+.2f}/{applied_right:+.2f}  loop_ms={loop_ms:.1f} "\n',
                '                    f"fps={fps_ema:.1f} pid={current_kp:.2f}/{current_ki:.2f}/{current_kd:.2f} "\n',
                '                    f"curve={curve_sharpness:.2f}/{curve_speed_scale:.2f}"\n',
                '                )\n'
            ])
        continue

    new_lines.append(line)

with open('main.py', 'w') as f:
    f.writelines(new_lines)
