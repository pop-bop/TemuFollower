import time
import cv2
from vision import VisionAgent
from control import ControlAgent
from hardware import RobotHardware

def main():
    print("Initializing Robot Modules...")

    hw = RobotHardware()
    vision = VisionAgent(resolution=(320, 240))
    control = ControlAgent(target_x_center=160, base_speed=1.0)

    print("Robot initialized. Starting main loop.")

    try:
        time.sleep(1.0)

        print("[Main] Starting continuous camera feed...")
        for frame in vision.get_frame():
            vision_data = vision.process_frame(frame)

            left_speed, right_speed = control.calculate_speeds(vision_data)

            print(f"[Main] State: {control.state} | Left: {left_speed:.2f} Right: {right_speed:.2f}")
            hw.set_speeds(left_speed, right_speed)

            # Indicator logic based on new state machine
            red_led = (control.state == "STOPPED")
            green_led = (control.state == "BACKTRACKING")
            buzzer = (vision_data.get("line_ended", False) or control.state == "SPINNING")
            hw.set_indicators(red_led=red_led, green_led=green_led, buzzer=buzzer)

            cv2.imshow("Robot Vision Live", frame)

            if vision_data["cnn_layers"] is not None:
                for layer_name, layer_data in vision_data["cnn_layers"].items():
                    cv2.imshow(f"Vision Layer: {layer_name}", layer_data)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[Main] 'q' pressed. Exiting loop.")
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        hw.stop()
        cv2.destroyAllWindows()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()
