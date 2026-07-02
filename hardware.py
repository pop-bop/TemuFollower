import time
import atexit

class RobotHardware:
    def __init__(self):
        """
        Initializes the robot hardware.
        Uses UART (GPIO 14, 15) for motors.
        Uses GPIO 5 (Red), 6 (Green), 13 (Buzzer) for indicators.
        """
        self.red_pin = 5
        self.green_pin = 6
        self.buzzer_pin = 13
        
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
            
            # Setup UART
            import serial
            try:
                self.uart = serial.Serial('/dev/serial0', 9600, timeout=1)
                self.has_uart = True
            except Exception as e:
                print(f"UART Error: {e}")
                self.has_uart = False
                
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

    def set_speeds(self, left_speed, right_speed):
        """
        Sets the speed for left and right motors.
        Speeds should be floats between 0.0 and 1.0.
        Outputs via UART on GPIO 14 (TX).
        """
        # Clamp values between 0.0 and 1.0
        left_speed = max(0.0, min(1.0, float(left_speed)))
        right_speed = max(0.0, min(1.0, float(right_speed)))
        
        msg = f"L{left_speed:.2f} R{right_speed:.2f}\n"
        print(f"[Hardware UART] Outputting: {msg.strip()}")
        
        if self.has_hardware and getattr(self, 'has_uart', False):
            try:
                self.uart.write(msg.encode('utf-8'))
            except Exception as e:
                print(f"UART Write Error: {e}")

    def stop(self):
        self.set_speeds(0.0, 0.0)
        self.set_indicators(False, False, False)

    def cleanup(self):
        print("Cleaning up hardware...")
        self.stop()
        if self.has_hardware:
            if getattr(self, 'has_uart', False):
                self.uart.close()
            self.GPIO.cleanup()
