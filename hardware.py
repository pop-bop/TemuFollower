import time
import atexit

class RobotHardware:
    def __init__(self):
        """
        Initializes the robot hardware.
        Drives an L298N dual H-bridge motor driver directly via GPIO
        (IN1-IN4 for direction, ENA/ENB PWM for speed).
        Uses GPIO 5 (Red), 6 (Green), 13 (Buzzer) for indicators.
        """
        self.red_pin = 5
        self.green_pin = 6
        self.buzzer_pin = 13

        self.LEFT_ENA = 12
        self.LEFT_IN1 = 16
        self.LEFT_IN2 = 20
        self.RIGHT_ENB = 18
        self.RIGHT_IN3 = 21
        self.RIGHT_IN4 = 26
        self.PWM_FREQUENCY_HZ = 1000

        self.has_hardware = False
        try:
            import RPi.GPIO as GPIO
            self.GPIO = GPIO
            self.has_hardware = True

            self.GPIO.setmode(self.GPIO.BCM)
            self.GPIO.setwarnings(False)

            # Setup Indicators
            self.GPIO.setup(self.red_pin, self.GPIO.OUT)
            self.GPIO.setup(self.green_pin, self.GPIO.OUT)
            self.GPIO.setup(self.buzzer_pin, self.GPIO.OUT)

            # Setup L298N motor driver
            self.GPIO.setup(self.LEFT_IN1, self.GPIO.OUT)
            self.GPIO.setup(self.LEFT_IN2, self.GPIO.OUT)
            self.GPIO.setup(self.RIGHT_IN3, self.GPIO.OUT)
            self.GPIO.setup(self.RIGHT_IN4, self.GPIO.OUT)
            self.GPIO.setup(self.LEFT_ENA, self.GPIO.OUT)
            self.GPIO.setup(self.RIGHT_ENB, self.GPIO.OUT)

            self.left_pwm = self.GPIO.PWM(self.LEFT_ENA, self.PWM_FREQUENCY_HZ)
            self.right_pwm = self.GPIO.PWM(self.RIGHT_ENB, self.PWM_FREQUENCY_HZ)
            self.left_pwm.start(0)
            self.right_pwm.start(0)

            atexit.register(self.cleanup)
        except ImportError:
            print("RPi.GPIO not found. Running in simulation/mock mode.")
            self.has_hardware = False

    def set_indicators(self, red_led=False, green_led=False, buzzer=False):
        if not self.has_hardware:
            return
        self.GPIO.output(self.red_pin, self.GPIO.HIGH if red_led else self.GPIO.LOW)
        self.GPIO.output(self.green_pin, self.GPIO.HIGH if green_led else self.GPIO.LOW)
        self.GPIO.output(self.buzzer_pin, self.GPIO.HIGH if buzzer else self.GPIO.LOW)

    def _set_motor(self, speed, in_a, in_b, pwm, invert=False):
        forward = speed >= 0
        if invert:
            forward = not forward
        if forward:
            self.GPIO.output(in_a, self.GPIO.HIGH)
            self.GPIO.output(in_b, self.GPIO.LOW)
        else:
            self.GPIO.output(in_a, self.GPIO.LOW)
            self.GPIO.output(in_b, self.GPIO.HIGH)
        pwm.ChangeDutyCycle(min(100.0, abs(speed) * 100.0))

    def set_speeds(self, left_speed, right_speed):
        """
        Sets the speed for left and right motors on the L298N driver.
        Speeds should be floats between -1.0 and 1.0.
        """
        # Clamp values between -1.0 and 1.0
        left_speed = max(-1.0, min(1.0, float(left_speed)))
        right_speed = max(-1.0, min(1.0, float(right_speed)))

        print(f"[Hardware L298N] L={left_speed:+.2f} R={right_speed:+.2f}")

        if self.has_hardware:
            self._set_motor(left_speed, self.LEFT_IN1, self.LEFT_IN2, self.left_pwm)
            # Right motor wiring is inverted compared to the left motor: forward
            # is IN3=LOW, IN4=HIGH (opposite of the left motor's pattern), so
            # invert=True flips the polarity here.
            self._set_motor(right_speed, self.RIGHT_IN3, self.RIGHT_IN4, self.right_pwm, invert=True)

    def stop(self):
        self.set_speeds(0.0, 0.0)
        self.set_indicators(False, False, False)

    def cleanup(self):
        print("Cleaning up hardware...")
        self.stop()
        if self.has_hardware:
            self.left_pwm.stop()
            self.right_pwm.stop()
            self.GPIO.cleanup()
