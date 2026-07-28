# TemuFollower Intel D435i + SPI Migration

This repository has been updated to support an Intel RealSense D435i camera (replacing the Raspberry Pi camera) and an SPI-based motor/LED controller (replacing direct GPIO control on the Pi).

## New Vision Pipeline
1. **RealSense Capture**: Uses `pyrealsense2` to fetch aligned color and depth streams. Intrinsics are dynamically pulled from the active profile.
2. **Perspective Warp**: Because the D435i is mounted at a 25-degree downward tilt, `vision.py` applies a homography transform (configured via `WARP_MATRIX` in `config.py`) to convert the angled view into a synthetic bird's-eye (top-down) view. 
3. **Obstacle Detection**: The depth stream is converted into a physical depth map and compared against a ground-plane expected depth map (calculated analytically using `CAMERA_TILT_ANGLE_DEG` and `CAMERA_MOUNT_HEIGHT_MM`). Any regions closer than `GROUND_DEPTH_TOLERANCE_MM` are flagged as obstacles (e.g. red blocks) and masked out of the flat red line tracking, naturally repelling the planned line-following path away from the obstacle.

## SPI Protocol (Pi -> Pico)
The Raspberry Pi acts as the SPI Master and no longer drives motors or LEDs directly. It relies on a Raspberry Pi Pico (SPI Slave) connected to SPI0 (`CE0`).

All packets begin with a 2-bit opcode and are transmitted msb-first:
- **`00` Movement**: Followed by five 16-bit vectors (x/turn, y/forward), spaced 0.2s apart. (Total: 82 bits, sent as 11 bytes). The Pico interpolates these into Bezier-smoothed motor PWM outputs.
- **`01` Interrupt**: No payload. Immediately flushes any queued motion and stops the robot. (Total: 2 bits, sent as 1 byte).
- **`10` Speciality**: Followed by 2 bits. Toggles an indicator for 0.5s. `00`=Green LED, `01`=Red LED, `10`=Buzzer. (Total: 4 bits, sent as 1 byte).
- **`11` Grabber**: Followed by 8 bits indicating the target angle for the grabber stepper motor. (Total: 10 bits, sent as 2 bytes).

*Note: All packets are padded with trailing zeros to complete full bytes for transmission via `spidev`.*
