from config import MOTOR_MIN_DUTY, MOTOR_DIR_FLIP_HYSTERESIS
from utils import clamp
from uart_master import UARTMaster, DIR_LEFT_REVERSE, DIR_RIGHT_REVERSE, BRAKE_LEFT, BRAKE_RIGHT

class RobotMotors:
    def __init__(self, uart_master: UARTMaster):
        self.uart = uart_master
        self._left_reversed = False
        self._right_reversed = False
        self._last_packet = None

    def _resolve_direction(self, value: float, was_reversed: bool) -> bool:
        # Only flip once the command clears the hysteresis band, so a wheel hovering
        # around zero doesn't toggle the H-bridge direction bit every frame.
        if was_reversed:
            return value <= MOTOR_DIR_FLIP_HYSTERESIS
        return value < -MOTOR_DIR_FLIP_HYSTERESIS

    def _apply(self, left: float, right: float, brake: bool = False):
        if brake:
            self._left_reversed = False
            self._right_reversed = False
            self._last_packet = None
            self.uart.set_motors(0, 0, BRAKE_LEFT | BRAKE_RIGHT)
            return

        left = clamp(left, -1.0, 1.0)
        right = clamp(right, -1.0, 1.0)

        # Hardware mapping: right motor is mirrored, so invert its direction
        # so that positive 'right' means forward movement.
        actual_right = -right

        self._left_reversed = self._resolve_direction(left, self._left_reversed)
        self._right_reversed = self._resolve_direction(actual_right, self._right_reversed)

        # Below MOTOR_MIN_DUTY the driver only buzzes the gearbox, so zero it out.
        left_mag = abs(left) if abs(left) >= MOTOR_MIN_DUTY else 0.0
        right_mag = abs(right) if abs(right) >= MOTOR_MIN_DUTY else 0.0

        left_duty = int(left_mag * 255.0)
        right_duty = int(right_mag * 255.0)

        dir_mask = 0
        if self._left_reversed:
            dir_mask |= DIR_LEFT_REVERSE
        if self._right_reversed:
            dir_mask |= DIR_RIGHT_REVERSE

        packet = (left_duty, right_duty, dir_mask)
        if packet == self._last_packet:
            return
        self._last_packet = packet

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
        self._left_reversed = False
        self._right_reversed = False
        self._last_packet = None
        self.uart.stop_all()

    def brake(self):
        self._apply(0.0, 0.0, brake=True)

    def drive(self, throttle: float, steering: float):
        left = throttle + steering
        right = throttle - steering
        self._apply(left, right)
