import time
import math
import random

class ControlAgent:
    def __init__(self, target_x_center=160, base_speed=1.0):
        self.target_x = target_x_center
        self.base_speed = base_speed # Punish for being slow -> default to max speed
        
        # Non-linear PID constants
        self.Kp = 0.003
        self.Ki = 0.0001
        self.Kd = 0.002
        self.nl_factor = 1.5 # Exponent for non-linear P gain
        
        self.integral = 0
        self.last_error = 0
        self.last_time = time.time()
        
        # State machine
        self.state = "FOLLOWING" # FOLLOWING, AVOIDING, RECOVERING, INTERSECTION
        self.last_known_error = 0 # To know which way to spin during recovery
        
        self.avoid_start_time = 0
        self.avoid_phase = 0
        
        self.intersection_start_time = 0
        self.intersection_dir = "left"
        
        self.saw_red_line_bottom = False
        self.saw_green_square_bottom = False
        
    def calculate_speeds(self, vision_data):
        """
        Takes vision_data dict and computes left/right motor speeds.
        Returns: (left_speed, right_speed) in range 0.0 to 1.0
        """
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        if dt <= 0:
            dt = 0.001
            
        # 1. State Machine Handling
        current_state = vision_data.get("special_state")
        
        # Memory updates for items passing underneath
        if current_state == "red_line_bottom":
            self.saw_red_line_bottom = True
        elif current_state == "green_square_bottom":
            self.saw_green_square_bottom = True
            
        # Trigger stop if we saw it underneath but now it's totally gone from vision
        if current_state not in ["red_line_bottom", "red_line_ahead"] and self.saw_red_line_bottom:
            self.saw_red_line_bottom = False
            self.state = "STOPPED_ON_RED"
            print("RED LINE PASSED UNDERNEATH (LOST FROM VISION)! Shutting down motors.")
            
        if current_state not in ["green_square_bottom", "green_square_ahead"] and self.saw_green_square_bottom:
            self.saw_green_square_bottom = False
            self.state = "STOPPED_ON_GREEN"
            print("GREEN SQUARE PASSED UNDERNEATH (LOST FROM VISION)! Shutting down motors.")

        if self.state == "STOPPED_ON_RED" or self.state == "STOPPED_ON_GREEN" or self.state == "STOPPED":
            return 0.0, 0.0
            
        if self.state == "AVOIDING":
            return self._handle_avoidance(current_time)
            
        if self.state == "INTERSECTION":
            return self._handle_intersection(current_time)
            
        if vision_data["obstacle_detected"]:
            self.state = "AVOIDING"
            self.avoid_start_time = current_time
            self.avoid_phase = 1
            print(f"Obstacle Cube detected! Initiating avoidance maneuver. (Line CX is still: {vision_data.get('line_center_x')})")
            return self._handle_avoidance(current_time)
            
        if vision_data.get("special_state") == "intersection":
            self.state = "INTERSECTION"
            self.intersection_start_time = current_time
            self.intersection_dir = random.choice(["left", "right"])
            print(f"Intersection detected! Randomly turning {self.intersection_dir.upper()}.")
            return self._handle_intersection(current_time)
            
        # 3. Check for line loss (Recovery)
        cx = vision_data.get("line_center_x")
        cy = vision_data.get("line_center_y")
        
        if cx is None:
            if vision_data["special_state"] == "gap":
                # Dashed line gap - maybe continue straight briefly before panicking
                pass # Fall through to recovery if gap persists too long
            self.state = "RECOVERING"
            return self._handle_recovery()
            
        # We have a valid line
        self.state = "FOLLOWING"
        
        # Calculate Error from the center of the frame (assuming 320 width)
        error = cx - self.target_x
        self.last_known_error = error
        
        # --- NON-LINEAR PID ---
        # Instead of linear P, we scale it non-linearly so larger errors cause exponentially sharper turns
        sign = 1 if error > 0 else -1
        p_error = sign * (abs(error) ** self.nl_factor)
        
        self.integral += error * dt
        # Anti-windup
        self.integral = max(-1000, min(1000, self.integral))
        
        derivative = (error - self.last_error) / dt
        self.last_error = error
        
        turn = (self.Kp * p_error) + (self.Ki * self.integral) + (self.Kd * derivative)
        
        # 4. Calculate speeds
        left_speed = self.base_speed
        right_speed = self.base_speed
        
        if turn > 0:
            right_speed -= turn
        else:
            left_speed += turn # turn is negative, so this reduces left_speed
            
        # Optional: Decelerate slightly on sharp turns
        if abs(error) > 50:
            left_speed *= 0.8
            right_speed *= 0.8
            
        print(f"[Control] State: FOLLOWING | Error: {error:.1f} | p_err: {p_error:.1f} | Turn: {turn:.2f}")
        return left_speed, right_speed
        
    def _handle_avoidance(self, current_time):
        """Hardcoded open-loop maneuver to bypass a red box."""
        elapsed = current_time - self.avoid_start_time
        
        # Phase 1: Turn sharp left (steer off line)
        if elapsed < 0.5:
            return 0.2, 0.8
        # Phase 2: Go straight ahead past the box
        elif elapsed < 1.5:
            return 0.8, 0.8
        # Phase 3: Turn sharp right (steer back toward line)
        elif elapsed < 2.0:
            return 0.8, 0.2
        else:
            # Resume looking for the line
            self.state = "RECOVERING" 
            return 0.5, 0.5

    def _handle_recovery(self):
        """Spin in place or sweep to find the line."""
        if self.last_known_error > 0:
            return 0.6, 0.0
        else:
            return 0.0, 0.6

    def _handle_intersection(self, current_time):
        """Hardcoded open-loop maneuver to turn left or right at an intersection."""
        elapsed = current_time - self.intersection_start_time
        
        # Drive forward slightly to position rotation center in the intersection
        if elapsed < 0.3:
            return 0.8, 0.8
        # Turn 90 degrees in the randomly chosen direction
        elif elapsed < 1.0:
            if self.intersection_dir == "left":
                return 0.2, 0.8
            else:
                return 0.8, 0.2
        else:
            # Resume looking for the line
            self.state = "RECOVERING"
            return 0.5, 0.5
