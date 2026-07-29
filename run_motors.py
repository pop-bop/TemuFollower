#!/usr/bin/env python3

from uart_master import UARTMaster
from motors import RobotMotors

speed = 50

def main():
    global speed

    uart = UARTMaster()
    motors = RobotMotors(uart)

    print("DPSI-LFR motor test (UART Mode)")
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
            try:
                cmd = input("> ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                break
            
            val = speed / 100.0

            if cmd == "f":
                motors.forward(val)
                print(f"forward, speed {speed}%")

            elif cmd == "b":
                motors.backward(val)
                print(f"backward, speed {speed}%")

            elif cmd == "l":
                motors.spin_left(val)
                print(f"spin left, speed {speed}%")

            elif cmd == "r":
                motors.spin_right(val)
                print(f"spin right, speed {speed}%")

            elif cmd == "s":
                motors.stop()
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
        motors.stop()
        uart.close()
        print("UART cleaned up")

if __name__ == "__main__":
    main()