"""Scratch: drive main()'s obstacle go-around with a synthetic approaching obstacle.

The dev box has no RealSense, so depth frames are synthesised: bare floor, then
an obstacle closing in, then depth dropout (the near-limit blind zone) which is
what triggers the manoeuvre.
"""
import sys, types, threading, time, math
import numpy as np

import config
config.SHOW_DEBUG_VIEW = False
config.LOOP_LOG_INTERVAL_S = 0.25

import camera, main as m

# There is no pigpio or motor hardware on the dev box, and create_motors() now
# raises rather than silently degrading, so hand main() a recording stub.
class FakeMotors:
    def __init__(self):
        self.commands = []

    def set_speeds(self, left, right):
        self.commands.append((left, right))

    def stop(self):
        self.commands.append(("stop",))

    def close(self):
        pass

fake_motors = FakeMotors()
m.create_motors = lambda: (fake_motors, fake_motors.close)

H, W = 480, 640
MOUNT, TILT, VFOV = 60.0, 25.0, 42.0
SCALE = 0.001

class FakeIntr:
    fx = fy = 465.6; ppx = 320.0; ppy = 240.0; width = W; height = H

m.get_intrinsics = lambda cam: FakeIntr()
m.has_imu = lambda cam: True

_ground = np.zeros((H, W), np.float32)
for y in range(H):
    ny = (y - H / 2) / (H / 2)
    a = max(math.radians(TILT + ny * VFOV / 2), 0.05)
    _ground[y, :] = MOUNT / math.sin(a)

def depth_with_obstacle(dist_mm):
    d = _ground.copy()
    x0, x1 = int(W * 0.35), int(W * 0.65)
    for y in range(H):
        ny = (y - H / 2) / (H / 2)
        a = math.radians(TILT + ny * VFOV / 2)
        wall = dist_mm / max(math.cos(a), 1e-3)
        gd = MOUNT / math.sin(max(a, 0.05))
        d[y, x0:x1] = np.where(wall < gd, wall, d[y, x0:x1])
    return np.clip(d / (SCALE * 1000.0), 0, 65535).astype(np.uint16)

BARE = np.clip(_ground / (SCALE * 1000.0), 0, 65535).astype(np.uint16)
BLIND = np.zeros((H, W), np.uint16)

G = 9.80665
tilt = math.radians(config.CAMERA_TILT_ANGLE_DEG)
ey, ez = -G * math.cos(tilt), -G * math.sin(tilt)
_t = [0.0]
_n = [0]

# A frame with a black line down the middle, so FOLLOW has something to track.
_line = np.full((H, W, 3), 200, np.uint8)
_line[:, int(W * 0.45):int(W * 0.55)] = 20
_blank = np.full((H, W, 3), 200, np.uint8)   # no line: forces REACQUIRE to run

def fake_read(kind, cam):
    i = _n[0]; _n[0] += 1
    # 0-14: bare floor. 15-44: obstacle closing 450->200mm. 45+: blind (dropout).
    if i < 15:
        depth = BARE
    elif i < 45:
        depth = depth_with_obstacle(450 - (i - 15) * 8)
    else:
        depth = BLIND
    # hide the line once the manoeuvre starts, so the robot must reacquire
    color = _blank if 47 <= i < 105 else _line
    motion = []
    for _ in range(8):
        _t[0] += 1 / 240.0
        motion.append(("accel", (0.0, ey, ez), _t[0]))
    return color, depth, motion

m.read_frame_with_motion = fake_read

# Force a healthy speed estimate so legs actually advance without real flow.
import odometry
_real_uv = odometry.SpeedEstimator.update_vision
def fake_uv(self, gray, dt, mask=None):
    self.speed_mps = 0.30
    self.source = "fused"
    return self.speed_mps
odometry.SpeedEstimator.update_vision = fake_uv

seen = []
_real_print = print
def stop_soon():
    time.sleep(22)
    import _thread
    _thread.interrupt_main()
threading.Thread(target=stop_soon, daemon=True).start()
m.main()
