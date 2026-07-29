

SOF = 0xAA

CMD_SET_MOTORS = 0x01
CMD_STOP_ALL = 0x02
CMD_SET_PWM_FREQ = 0x03
CMD_STATUS_RESP = 0x11
CMD_PING = 0xFF

def calc_checksum(data: bytes) -> int:
    ck = 0
    for b in data:
        ck ^= b
    return ck

def build_packet(cmd_id: int, payload: bytes = b"") -> bytes:
    length = len(payload)
    packet_no_chk = bytes([SOF, length, cmd_id]) + payload
    chk = calc_checksum(packet_no_chk)
    return packet_no_chk + bytes([chk])
