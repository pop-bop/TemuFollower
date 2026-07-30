"""Pi-direct motor backend for the IBT-2 / BTS7960 half-bridge boards.

Replaces the ESP32 UART hop. Each board takes an LPWM/RPWM pair: to drive a
side you PWM one input and hold the other at zero. Driving both HIGH turns on
both high-side FETs, which on many clone boards is a shoot-through path, so
brake is implemented as both-LOW.
"""

import threading
import time

from config import (
    LEFT_LPWM, LEFT_RPWM, RIGHT_LPWM, RIGHT_RPWM,
    LEFT_EN, RIGHT_EN, PWM_FREQUENCY_HZ, MOTOR_COMMAND_TIMEOUT_S,
    MOTOR_MIN_DUTY, MOTOR_DIR_FLIP_HYSTERESIS,
    LEFT_MOTOR_INVERT, RIGHT_MOTOR_INVERT,
)
from utils import clamp

try:
    import pigpio
except ImportError:
    pigpio = None

_PWM_RANGE = 255


import atexit
class PiMotorDriver:
    """Low-level pin driver. Owns the pigpio handle and the failsafe watchdog."""

    def __init__(self):
        if pigpio is None:
            # python3-pigpio installs to system dist-packages, which a venv
            # cannot see -- inside one you need the pip client as well.
            raise RuntimeError(
                "pigpio module not importable. System-wide: sudo apt install python3-pigpio. "
                "In a venv: pip install pigpio"
            )

        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("Cannot reach pigpiod. Start it with: sudo systemctl start pigpiod")

        self._pins = (LEFT_LPWM, LEFT_RPWM, RIGHT_LPWM, RIGHT_RPWM)
        for pin in self._pins:
            self.pi.set_mode(pin, pigpio.OUTPUT)
            actual = self.pi.set_PWM_frequency(pin, PWM_FREQUENCY_HZ)
            if actual != PWM_FREQUENCY_HZ:
                print(f"[WARN] GPIO {pin}: asked {PWM_FREQUENCY_HZ}Hz, got {actual}Hz. "
                      f"Duty resolution is {self.pi.get_PWM_real_range(pin)} steps.")
            self.pi.set_PWM_range(pin, _PWM_RANGE)
            self._write(pin, 0)

        for pin in (LEFT_EN, RIGHT_EN):
            if pin is not None:
                self.pi.set_mode(pin, pigpio.OUTPUT)
                self.pi.write(pin, 1)

        self._lock = threading.Lock()
        self._last_command = time.monotonic()
        self._tripped = False
        self._closed = False
        self._watchdog = threading.Thread(target=self._watch, daemon=True)
        self._watchdog.start()
        atexit.register(self.close)

    def _write(self, pin, duty):
        """duty is 0.._PWM_RANGE.

        Software (DMA-timed) PWM, not hardware_PWM: the Pi has only two hardware
        channels and both motor pin pairs land on them (12/18 -> ch0, 13/19 ->
        ch1). Since a channel's duty is shared by every GPIO on it, the last
        write won and the left side's command was silently discarded. Software
        PWM is independent per GPIO.
        """
        result = self.pi.set_PWM_dutycycle(pin, duty)
        if result < 0:
            raise RuntimeError(f"pigpio rejected duty {duty} on GPIO {pin}: {result}")

    def _apply_side(self, lpwm_pin, rpwm_pin, duty, reverse):
        if reverse:
            self._write(lpwm_pin, 0)
            self._write(rpwm_pin, duty)
        else:
            self._write(lpwm_pin, duty)
            self._write(rpwm_pin, 0)

    def set_outputs(self, left_duty, right_duty, left_reverse, right_reverse):
        with self._lock:
            if self._closed:
                return
            self._apply_side(LEFT_LPWM, LEFT_RPWM, left_duty, left_reverse)
            self._apply_side(RIGHT_LPWM, RIGHT_RPWM, right_duty, right_reverse)
            self._last_command = time.monotonic()
            self._tripped = False

    def coast(self):
        with self._lock:
            if self._closed:
                return
            for pin in self._pins:
                self._write(pin, 0)
            self._last_command = time.monotonic()
            self._tripped = False

    def brake(self):
        # Both inputs LOW shorts the winding through the low-side FETs.
        self.coast()

    def _watch(self):
        while True:
            time.sleep(MOTOR_COMMAND_TIMEOUT_S / 3.0)
            try:
                with self._lock:
                    if self._closed:
                        return
                    if self._tripped:
                        continue
                    if time.monotonic() - self._last_command <= MOTOR_COMMAND_TIMEOUT_S:
                        continue
                    for pin in self._pins:
                        self._write(pin, 0)
                    self._tripped = True
            except Exception as e:
                # Never let a write error kill this thread -- it is the only
                # thing that cuts the motors if the control loop stops feeding
                # commands, and a dead watchdog fails silently and dangerously.
                print(f"[SAFE] watchdog write failed: {e}")
                continue
            print("[SAFE] motor command timeout - outputs cut")

    @property
    def failsafe_tripped(self):
        return self._tripped

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                for pin in self._pins:
                    self._write(pin, 0)
                for pin in (LEFT_EN, RIGHT_EN):
                    if pin is not None:
                        self.pi.write(pin, 0)
            finally:
                # Dropping the daemon connection releases the pins regardless, so
                # this must run even if zeroing a duty failed.
                self.pi.stop()


class PiRobotMotors:
    """Maps signed left/right commands onto the two bridges."""

    def __init__(self, driver=None):
        self.driver = driver if driver is not None else PiMotorDriver()
        self._left_reversed = False
        self._right_reversed = False
        self._last = None

    def _resolve_direction(self, value, was_reversed):
        # Only flip once the command clears the hysteresis band, so a wheel
        # hovering near zero doesn't toggle bridge direction every frame.
        if was_reversed:
            return value <= MOTOR_DIR_FLIP_HYSTERESIS
        return value < -MOTOR_DIR_FLIP_HYSTERESIS

    def _apply(self, left, right, brake=False):
        if brake:
            self._left_reversed = False
            self._right_reversed = False
            self._last = None
            self.driver.brake()
            return

        left = clamp(left, -1.0, 1.0)
        right = clamp(right, -1.0, 1.0)

        # Apply wiring polarity to the signed value, before anything is derived
        # from it. The previous code negated only the copy used for the
        # direction bit while the magnitude kept the original sign, so a side
        # ran backwards at the right speed: forward and turn came out swapped.
        if LEFT_MOTOR_INVERT:
            left = -left
        if RIGHT_MOTOR_INVERT:
            right = -right

        self._left_reversed = self._resolve_direction(left, self._left_reversed)
        self._right_reversed = self._resolve_direction(right, self._right_reversed)

        # Below MOTOR_MIN_DUTY the bridge only buzzes the gearbox.
        left_mag = abs(left) if abs(left) >= MOTOR_MIN_DUTY else 0.0
        right_mag = abs(right) if abs(right) >= MOTOR_MIN_DUTY else 0.0

        packet = (
            int(left_mag * 255.0), int(right_mag * 255.0),
            self._left_reversed, self._right_reversed,
        )
        # Unlike the UART backend there is no reason to suppress duplicate
        # commands -- there is no bus to spare, and every call has to reach the
        # driver anyway to refresh the failsafe watchdog.
        self._last = packet
        self.driver.set_outputs(*packet)

    def set_speeds(self, left, right):
        self._apply(left, right)

    def forward(self, speed):
        self._apply(abs(speed), abs(speed))

    def backward(self, speed):
        self._apply(-abs(speed), -abs(speed))

    def turn_left(self, speed):
        self._apply(0.0, abs(speed))

    def turn_right(self, speed):
        self._apply(abs(speed), 0.0)

    def spin_left(self, speed):
        self._apply(-abs(speed), abs(speed))

    def spin_right(self, speed):
        self._apply(abs(speed), -abs(speed))

    def stop(self):
        self._left_reversed = False
        self._right_reversed = False
        self._last = None
        self.driver.coast()

    def brake(self):
        self._apply(0.0, 0.0, brake=True)

    def drive(self, throttle, steering):
        self._apply(throttle + steering, throttle - steering)

    def close(self):
        self.driver.close()


def create_motors():
    """Build the motor backend.

    Returns (motors, closer). Raises rather than degrading: an earlier version
    fell back to the ESP32 UART backend, whose transport went silent when the
    port was absent, so the robot ran a full course with dead motors while the
    control loop logged healthy commands.
    """
    motors = PiRobotMotors()
    print("Motors: Pi GPIO backend (pigpio)")
    return motors, motors.close
