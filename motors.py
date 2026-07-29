from utils import clamp
from uart_master import UARTMaster, DIR_LEFT_REVERSE, DIR_RIGHT_REVERSE, BRAKE_LEFT, BRAKE_RIGHT

class RobotMotors:
    def __init__(self, uart_master: UARTMaster):
        self.uart = uart_master

    def _apply(self, left: float, right: float, brake: bool = False):
        if brake:
            self.uart.set_motors(0, 0, BRAKE_LEFT | BRAKE_RIGHT)
            return

        left = clamp(left, -1.0, 1.0)
        right = clamp(right, -1.0, 1.0)

        left_duty = int(abs(left) * 255.0)
        right_duty = int(abs(right) * 255.0)

        dir_mask = 0
        if left < 0:
            dir_mask |= DIR_LEFT_REVERSE
        if right < 0:
            dir_mask |= DIR_RIGHT_REVERSE

        self.uart.set_motors(left_duty, right_duty, dir_mask)

    def set_speeds(self, left: float, right: float):
        self._apply(left, right)

    def forward(self, speed: float):
        self._apply(abs(speed), abs(speed))

    def backward(self, speed: float):
        self._apply(-abs(speed), -abs(speed))

    def turn_left(self, speed: float):
        # Pivot left: left motor stopped or slower, right motor forward
        self._apply(0.0, abs(speed))

    def turn_right(self, speed: float):
        self._apply(abs(speed), 0.0)

    def spin_left(self, speed: float):
        self._apply(-abs(speed), abs(speed))

    def spin_right(self, speed: float):
        self._apply(abs(speed), -abs(speed))

    def stop(self):
        self.uart.stop_all()

    def brake(self):
        self._apply(0.0, 0.0, brake=True)

    def drive(self, throttle: float, steering: float):
        left = throttle + steering
        right = throttle - steering
        self._apply(left, right)
