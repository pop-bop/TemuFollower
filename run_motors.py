#!/usr/bin/env python3

from spi import SPIController, pack_movement, pack_interrupt
from config import SPI_BUS, SPI_DEVICE, SPI_MAX_SPEED_HZ

speed = 50

def send_movement(spi, turn_val, fwd_val):
    # Pack exactly 5 waypoints for SPI (constant movement)
    waypoints = [(turn_val, fwd_val)] * 5
    spi.transmit(pack_movement(waypoints))

def main():
    global speed

    spi = SPIController(bus=SPI_BUS, device=SPI_DEVICE, max_speed_hz=SPI_MAX_SPEED_HZ)

    print("DPSI-LFR motor test (SPI Mode)")
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
            
            fwd_val = int((speed / 100.0) * 127.0)
            turn_val = int((speed / 100.0) * 127.0)

            if cmd == "f":
                send_movement(spi, 0, fwd_val)
                print(f"forward, speed {speed}%")

            elif cmd == "b":
                send_movement(spi, 0, -fwd_val)
                print(f"backward, speed {speed}%")

            elif cmd == "l":
                send_movement(spi, turn_val, 0)
                print(f"spin left, speed {speed}%")

            elif cmd == "r":
                send_movement(spi, -turn_val, 0)
                print(f"spin right, speed {speed}%")

            elif cmd == "s":
                spi.transmit(pack_interrupt())
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
        spi.transmit(pack_interrupt())
        print("SPI cleaned up")

if __name__ == "__main__":
    main()