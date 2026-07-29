import serial
import time
from protocol import build_packet, CMD_SET_MOTORS, CMD_STOP_ALL, CMD_PING

DIR_LEFT_REVERSE = 0x01
DIR_RIGHT_REVERSE = 0x02
BRAKE_LEFT = 0x04
BRAKE_RIGHT = 0x08

class UARTMaster:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200, timeout=1.0):
        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
            time.sleep(2) # Wait for ESP32 to reboot after serial connection
        except Exception as e:
            print(f"UART connection failed: {e}")
            self.ser = None

    def send_packet(self, cmd_id: int, payload: bytes = b""):
        if self.ser is None:
            return
        packet = build_packet(cmd_id, payload)
        try:
            self.ser.write(packet)
            self.ser.flush()
        except serial.SerialException as e:
            print(f"UART write failed: {e}")
            self.ser = None

    def set_motors(self, left_duty: int, right_duty: int, dir_mask: int):
        payload = bytes([left_duty & 0xFF, right_duty & 0xFF, dir_mask & 0xFF])
        self.send_packet(CMD_SET_MOTORS, payload)

    def stop_all(self):
        self.send_packet(CMD_STOP_ALL)

    def ping(self):
        self.send_packet(CMD_PING, b"\x55")

    def close(self):
        if self.ser:
            self.ser.close()
