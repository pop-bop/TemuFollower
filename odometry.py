"""Speed estimation by fusing the D435i IMU with BEV visual odometry.

The robot has no wheel encoders, so absolute speed has to be inferred. Two
sources with opposite failure modes are combined:

  * IMU accelerometer -- ~250Hz, low noise, but integrating it accumulates
    unbounded drift within seconds.
  * BEV optical flow -- ~30Hz, noisy, but drift-free and absolute. Because the
    bird's-eye image has a known mm/px scale, flow is metric with no monocular
    scale ambiguity.

A complementary filter runs the fast IMU integration and lets flow continuously
reset its drift. Speed matters because it determines how long ground seen in the
near ROI takes to reach the wheels -- without it there is no principled vision
delay.
"""

import math
import time

import cv2
import numpy as np

from config import (
    ODOMETRY_ALPHA, ODOMETRY_MIN_FLOW_PX, ODOMETRY_MAX_CORNERS,
    ODOMETRY_QUALITY, ODOMETRY_MIN_DISTANCE_PX, ODOMETRY_MIN_TRACKED,
    ODOMETRY_MAX_SPEED_MPS, BEV_MM_PER_PX,
    IMU_MAX_PITCH_DEV_DEG, CAMERA_TILT_ANGLE_DEG,
)
from utils import clamp

_GRAVITY = 9.80665

_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


class VisualOdometry:
    """Forward speed from optical flow in the rectified ground plane.

    Only the median forward component is used. Lateral flow is dominated by
    yaw rotation, which the gyro measures far better.
    """

    def __init__(self):
        self.prev_gray = None
        self.prev_points = None
        self.last_speed_mps = 0.0
        self.last_confidence = 0.0

    def reset(self):
        self.prev_gray = None
        self.prev_points = None
        self.last_speed_mps = 0.0
        self.last_confidence = 0.0

    def _detect(self, gray, mask):
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=ODOMETRY_MAX_CORNERS,
            qualityLevel=ODOMETRY_QUALITY,
            minDistance=ODOMETRY_MIN_DISTANCE_PX,
            mask=mask,
        )

    def update(self, bev_gray, dt, valid_mask=None):
        """Returns (speed_mps, confidence). Confidence 0 means unusable."""
        if dt <= 1e-6:
            return self.last_speed_mps, 0.0

        if self.prev_gray is None:
            self.prev_gray = bev_gray
            self.prev_points = self._detect(bev_gray, valid_mask)
            return 0.0, 0.0

        if self.prev_points is None or len(self.prev_points) < ODOMETRY_MIN_TRACKED:
            self.prev_points = self._detect(self.prev_gray, valid_mask)
            if self.prev_points is None or len(self.prev_points) < ODOMETRY_MIN_TRACKED:
                self.prev_gray = bev_gray
                self.last_confidence = 0.0
                return self.last_speed_mps, 0.0

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, bev_gray, self.prev_points, None, **_LK_PARAMS
        )
        if next_points is None or status is None:
            self.prev_gray = bev_gray
            self.prev_points = None
            self.last_confidence = 0.0
            return self.last_speed_mps, 0.0

        keep = status.ravel() == 1
        good_prev = self.prev_points[keep].reshape(-1, 2)
        good_next = next_points[keep].reshape(-1, 2)

        self.prev_gray = bev_gray
        self.prev_points = (
            good_next.reshape(-1, 1, 2) if len(good_next) >= ODOMETRY_MIN_TRACKED
            else self._detect(bev_gray, valid_mask)
        )

        if len(good_next) < ODOMETRY_MIN_TRACKED:
            self.last_confidence = 0.0
            return self.last_speed_mps, 0.0

        # Ground moves DOWN the BEV image (+row) as the robot moves forward,
        # because row 0 is the far edge.
        dy = good_next[:, 1] - good_prev[:, 1]
        median_dy = float(np.median(dy))

        # Agreement among tracked points is the confidence signal: real
        # translation moves them together, noise does not.
        spread = float(np.median(np.abs(dy - median_dy)))
        confidence = clamp(1.0 - spread / 3.0, 0.0, 1.0)
        confidence *= clamp(len(good_next) / float(ODOMETRY_MIN_TRACKED * 3), 0.0, 1.0)

        if abs(median_dy) < ODOMETRY_MIN_FLOW_PX:
            self.last_speed_mps = 0.0
            self.last_confidence = confidence
            return 0.0, confidence

        speed_mps = (median_dy * BEV_MM_PER_PX / 1000.0) / dt

        if abs(speed_mps) > ODOMETRY_MAX_SPEED_MPS:
            self.last_confidence = 0.0
            return self.last_speed_mps, 0.0

        self.last_speed_mps = speed_mps
        self.last_confidence = confidence
        return speed_mps, confidence


class ImuIntegrator:
    """Forward-velocity integration from D435i accel samples.

    The D435i IMU is not hardware-synced to the frames, so samples are
    integrated on their own timestamps rather than on frame dt.
    """

    def __init__(self):
        self.velocity_mps = 0.0
        self.last_ts = None
        self.pitch_dev_deg = 0.0
        self.yaw_rate_dps = 0.0

    def reset(self, velocity_mps=0.0):
        self.velocity_mps = velocity_mps
        self.last_ts = None

    def add_gyro(self, gyro_xyz):
        # D435i gyro: y is yaw about the vertical axis.
        self.yaw_rate_dps = math.degrees(gyro_xyz[1])

    def add_accel(self, accel_xyz, timestamp_s):
        """Integrate one accel sample. Returns the running velocity estimate."""
        if self.last_ts is None:
            self.last_ts = timestamp_s
            return self.velocity_mps

        dt = timestamp_s - self.last_ts
        self.last_ts = timestamp_s
        # Out-of-order or duplicate samples do happen on the D435i.
        if dt <= 0.0 or dt > 0.5:
            return self.velocity_mps

        ax, ay, az = accel_xyz

        tilt = math.radians(CAMERA_TILT_ANGLE_DEG)
        # An accelerometer reports specific force, not the gravity field, so at
        # rest it reads the UP vector: a level D435i gives (0, -g, 0) with +Y
        # down. Both components are negative once the lens is pitched down.
        expected_y = -_GRAVITY * math.cos(tilt)
        expected_z = -_GRAVITY * math.sin(tilt)

        # Ground-forward direction expressed in the tilted camera frame.
        fwd_y = -math.sin(tilt)
        fwd_z = math.cos(tilt)

        # How far measured gravity has swung from the mounted attitude. Ramps and
        # speed bumps pitch the chassis and leak gravity into the forward axis,
        # so those samples must not be integrated.
        magnitude = math.sqrt(ax * ax + ay * ay + az * az)
        if magnitude > 1e-6:
            cos_dev = clamp(
                (ay * expected_y + az * expected_z) / (magnitude * _GRAVITY),
                -1.0, 1.0,
            )
            self.pitch_dev_deg = math.degrees(math.acos(cos_dev))
        else:
            self.pitch_dev_deg = 90.0

        if self.pitch_dev_deg > IMU_MAX_PITCH_DEV_DEG:
            return self.velocity_mps

        # Project the residual onto ground-forward. Taking the Z residual alone
        # would drop a factor of cos(tilt) and under-read accel by ~9%.
        forward_accel = (ay - expected_y) * fwd_y + (az - expected_z) * fwd_z
        self.velocity_mps += forward_accel * dt
        return self.velocity_mps


class SpeedEstimator:
    """Complementary filter over ImuIntegrator and VisualOdometry."""

    def __init__(self):
        self.imu = ImuIntegrator()
        self.vo = VisualOdometry()
        self.speed_mps = 0.0
        self.source = "init"

    def reset(self):
        self.imu.reset()
        self.vo.reset()
        self.speed_mps = 0.0
        self.source = "reset"

    def add_imu_accel(self, accel_xyz, timestamp_s):
        self.speed_mps = self.imu.add_accel(accel_xyz, timestamp_s)

    def add_imu_gyro(self, gyro_xyz):
        self.imu.add_gyro(gyro_xyz)

    def update_vision(self, bev_gray, dt, valid_mask=None):
        """Fold in a flow measurement. Call once per frame."""
        flow_speed, confidence = self.vo.update(bev_gray, dt, valid_mask)

        if confidence <= 0.0:
            self.source = "imu"
            return self.speed_mps

        # Trust flow more as its confidence rises. Flow is the only drift-free
        # term, so it must always retain some authority.
        alpha = ODOMETRY_ALPHA * (1.0 - confidence) + (1.0 - ODOMETRY_ALPHA) * confidence
        alpha = clamp(alpha, 0.0, ODOMETRY_ALPHA)

        fused = alpha * self.speed_mps + (1.0 - alpha) * flow_speed
        self.speed_mps = fused
        # Rewind the integrator to the fused value, which is what makes flow
        # reset IMU drift rather than merely being averaged against it.
        self.imu.velocity_mps = fused
        self.source = "fused"
        return fused

    def fallback_from_command(self, applied_forward, top_speed_mps=1.2):
        """Open-loop guess for when odometry is unhealthy."""
        self.speed_mps = abs(applied_forward) * top_speed_mps
        self.source = "command"
        return self.speed_mps

    @property
    def confidence(self):
        return self.vo.last_confidence

    @property
    def yaw_rate_dps(self):
        return self.imu.yaw_rate_dps
