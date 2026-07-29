

try:
    import spidev
except ImportError:
    spidev = None

def init_spi(bus=0, device=0, max_speed_hz=1000000):
    if spidev is None:
        print("spidev not found (running off-Pi?)")
        return None
    spi = spidev.SpiDev()
    spi.open(bus, device)
    spi.max_speed_hz = max_speed_hz
    spi.mode = 0
    return spi

def pack_movement(waypoints) -> bytes:
    """
    Opcode 00. Followed by 80 bits = 5 vectors, each vector is 16 bits (2 components * 8 bits each).
    Each component is a signed 8-bit integer (-128 to +127).
    Total bits: 2 + 80 = 82 bits.
    Pad to 11 bytes (88 bits) by shifting left by 6 bits.
    """
    assert len(waypoints) == 5, "Movement packet requires exactly 5 waypoints"
    val = 0
    opcode = 0b00
    val |= (opcode & 0x3)
    
    for x, y in waypoints:
        x_int = max(-128, min(127, int(x)))
        if x_int < 0:
            x_int = (1 << 8) + x_int
        
        y_int = max(-128, min(127, int(y)))
        if y_int < 0:
            y_int = (1 << 8) + y_int
            
        val = (val << 8) | (x_int & 0xFF)
        val = (val << 8) | (y_int & 0xFF)
        
    val = val << 6
    return val.to_bytes(11, 'big')

def pack_interrupt() -> bytes:
    """
    Opcode 01. No payload. Total 2 bits. Pad to 1 byte (8 bits).
    """
    val = (0b01 << 6)
    return val.to_bytes(1, 'big')

def pack_speciality(code: int) -> bytes:
    """
    Opcode 10. Followed by 2 bits. Total 4 bits. Pad to 1 byte (8 bits).
    00: Green LED on for 0.5s
    01: Red LED on for 0.5s
    10: Buzzer on for 0.5s
    11: Reserved
    """
    val = (0b10 << 6) | ((code & 0x3) << 4)
    return val.to_bytes(1, 'big')

def pack_grabber(angle: int) -> bytes:
    """
    Opcode 11. Followed by 8 bits. Total 10 bits. Pad to 2 bytes (16 bits).
    """
    val = (0b11 << 14) | ((angle & 0xFF) << 6)
    return val.to_bytes(2, 'big')

class SPIController:
    def __init__(self, bus=0, device=0, max_speed_hz=1000000):
        self.spi = init_spi(bus, device, max_speed_hz)
        self.mock_mode = self.spi is None
        
    def transmit(self, data: bytes):
        if self.mock_mode:
            # print(f"Mock SPI Transmit: {data.hex()}")
            return
        self.spi.xfer2(list(data))
