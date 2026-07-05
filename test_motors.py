#!/usr/bin/env python3

import RPi.GPIO as GPIO

LEFT_ENA = 12
LEFT_IN1 = 16
LEFT_IN2 = 20

RIGHT_ENB = 18
RIGHT_IN3 = 21
RIGHT_IN4 = 26

PWM_FREQUENCY_HZ = 1000

speed = 50


def setup():
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


def set_left(direction, left_pwm):
    if direction == "forward":
        GPIO.output(LEFT_IN1, GPIO.HIGH)
        GPIO.output(LEFT_IN2, GPIO.LOW)
        left_pwm.ChangeDutyCycle(speed)

    elif direction == "backward":
        GPIO.output(LEFT_IN1, GPIO.LOW)
        GPIO.output(LEFT_IN2, GPIO.HIGH)
        left_pwm.ChangeDutyCycle(speed)

    else:
        GPIO.output(LEFT_IN1, GPIO.LOW)
        GPIO.output(LEFT_IN2, GPIO.LOW)
        left_pwm.ChangeDutyCycle(0)


def set_right(direction, right_pwm):
    if direction == "forward":
        GPIO.output(RIGHT_IN3, GPIO.LOW)
        GPIO.output(RIGHT_IN4, GPIO.HIGH)
        right_pwm.ChangeDutyCycle(speed)

    elif direction == "backward":
        GPIO.output(RIGHT_IN3, GPIO.HIGH)
        GPIO.output(RIGHT_IN4, GPIO.LOW)
        right_pwm.ChangeDutyCycle(speed)

    else:
        GPIO.output(RIGHT_IN3, GPIO.LOW)
        GPIO.output(RIGHT_IN4, GPIO.LOW)
        right_pwm.ChangeDutyCycle(0)


def stop(left_pwm, right_pwm):
    set_left("stop", left_pwm)
    set_right("stop", right_pwm)


def main():
    global speed

    left_pwm, right_pwm = setup()

    print("DPSI-LFR motor test")
    print("Commands:")
    print("f = forward")
    print("b = backward")
    print("l = spin left")
    print("r = spin right")
    print("s = stop")
    print("+ = faster")
    print("- = slower")
    print("q = quit")
    print(f"Current speed: {speed}%")

    try:
        while True:
            cmd = input("> ").strip().lower()

            if cmd == "f":
                set_left("forward", left_pwm)
                set_right("forward", right_pwm)
                print(f"forward, speed {speed}%")

            elif cmd == "b":
                set_left("backward", left_pwm)
                set_right("backward", right_pwm)
                print(f"backward, speed {speed}%")

            elif cmd == "l":
                set_left("backward", left_pwm)
                set_right("forward", right_pwm)
                print(f"spin left, speed {speed}%")

            elif cmd == "r":
                set_left("forward", left_pwm)
                set_right("backward", right_pwm)
                print(f"spin right, speed {speed}%")

            elif cmd == "s":
                stop(left_pwm, right_pwm)
                print("stopped")

            elif cmd == "+":
                speed = min(100, speed + 10)
                print(f"speed now {speed}%")

            elif cmd == "-":
                speed = max(0, speed - 10)
                print(f"speed now {speed}%")

            elif cmd == "q":
                print("quitting")
                break

            else:
                print("unknown command")

    finally:
        stop(left_pwm, right_pwm)
        left_pwm.stop()
        right_pwm.stop()
        GPIO.cleanup()
        print("GPIO cleaned up")


if __name__ == "__main__":
    main()