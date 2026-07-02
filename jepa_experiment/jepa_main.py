import cv2
import time
import sys
import os
import torch
import numpy as np

# Add the parent directory to sys.path so we can import hardware
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hardware import RobotHardware
from jepa import LineJEPA

def main():
    print("Initializing JEPA Robot Modules...")
    hw = RobotHardware()
    
    # Initialize Camera
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    # Initialize PyTorch JEPA Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LineJEPA().to(device)
    
    weights_path = os.path.join(os.path.dirname(__file__), "jepa_weights.pth")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"JEPA weights loaded on {device}")
    else:
        print(f"WARNING: JEPA weights not found. Using untrained random weights. Run train_jepa.py first!")
    model.eval()

    print("[JEPA Main] Starting continuous camera feed...")
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                break
                
            h, w = frame.shape[:2]
            roi = frame[int(h/3):, :]
            
            # Prepare tensor for JEPA
            tensor_img = cv2.resize(roi, (160, 120))
            tensor_img = cv2.cvtColor(tensor_img, cv2.COLOR_BGR2RGB)
            tensor_img = tensor_img.astype(np.float32) / 255.0
            tensor_img = np.transpose(tensor_img, (2, 0, 1))
            tensor_img = np.expand_dims(tensor_img, axis=0)
            
            input_tensor = torch.tensor(tensor_img).to(device)
            
            with torch.no_grad():
                embedding, state = model(input_tensor)
                
            pos = state[0, 0].item()    # -1.0 to 1.0
            angle = state[0, 1].item()  # radians
            
            # Control Logic (PID on abstract state)
            error = (pos * 1.0) + (angle * 0.5)
            
            # Simple PID variables
            Kp = 3.0
            turn = Kp * error
            
            left_speed = 1.0
            right_speed = 1.0
            if turn > 0:
                right_speed -= turn
            else:
                left_speed += turn
                
            # Hardware execution
            hw.set_speeds(left_speed, right_speed)
            
            # Visualization
            center_x = int((pos + 1.0) / 2.0 * w)
            center_y = int(h/3) + int(roi.shape[0] / 2)
            dx = int(60 * np.sin(angle))
            dy = int(60 * np.cos(angle))
            
            cv2.circle(frame, (center_x, center_y), 8, (0, 255, 0), -1)
            cv2.line(frame, (center_x, center_y), (center_x + dx, center_y - dy), (255, 0, 255), 4)
            cv2.putText(frame, f"JEPA Pos: {pos:.2f} Ang: {angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            cv2.imshow("JEPA Robot Vision", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nShutdown signal received.")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        hw.cleanup()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()
