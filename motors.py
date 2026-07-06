import RPi.GPIO as GPIO

from config import (
    LEFT_ENA, LEFT_IN1, LEFT_IN2, RIGHT_ENB, RIGHT_IN3, RIGHT_IN4,
    PWM_FREQUENCY_HZ,
)
from utils import clamp


def setup_motors():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    GPIO.setup(LEFT_IN1, GPIO.OUT)
    GPIO.setup(LEFT_IN2, GPIO.OUT)
    GPIO.setup(RIGHT_IN3, GPIO.OUT)
    GPIO.setup(RIGHT_IN4, GPIO.OUT)
    GPIO.setup(LEFT_ENA, GPIO.OUT)
    GPIO.setup(RIGHT_ENB, GPIO.OUT)

    left_pwm = GPIO.PWM(LEFT_ENA, PWM_FREQUENCY_HZ)
    right_pwm = GPIO.PWM(RIGHT_ENB, PWM_FREQUENCY_HZ)
    left_pwm.start(0)
    right_pwm.start(0)
    return left_pwm, right_pwm


def set_motor_speed(speed, in_a, in_b, pwm, invert=False):
    forward = speed >= 0
    if invert:
        forward = not forward
    if forward:
        GPIO.output(in_a, GPIO.HIGH)
        GPIO.output(in_b, GPIO.LOW)
    else:
        GPIO.output(in_a, GPIO.LOW)
        GPIO.output(in_b, GPIO.HIGH)
    pwm.ChangeDutyCycle(clamp(abs(speed), 0.0, 1.0) * 100.0)


def set_speeds(left, right, left_pwm, right_pwm):
    left = clamp(left, -1.0, 1.0)
    right = clamp(right, -1.0, 1.0)
    set_motor_speed(left, LEFT_IN1, LEFT_IN2, left_pwm)
    set_motor_speed(right, RIGHT_IN3, RIGHT_IN4, right_pwm, invert=True)


def slew_toward(current, target, max_step):
    if target > current:
        stepped = min(target, current + max_step)
    elif target < current:
        stepped = max(target, current - max_step)
    else:
        return current

    # Force a stop before reversing direction, even if max_step is large
    # enough (e.g. from a slow/stuttered frame) to jump straight past zero
    # into a fast reverse in a single step.
    if (current > 0.0 and stepped < 0.0) or (current < 0.0 and stepped > 0.0):
        return 0.0
    return stepped


def stop_motors(left_pwm, right_pwm):
    set_speeds(0.0, 0.0, left_pwm, right_pwm)
