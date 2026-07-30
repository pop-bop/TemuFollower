# PID
KP = 0.95
KI = 0.01
KD = 0.15
TURN_LIMIT = 0.80
CENTER_DEADZONE = 0.12
ADAPTIVE_PID_ENABLED = True
ADAPTIVE_KP_ERROR_BOOST = 0.55
ADAPTIVE_KP_CONFIDENCE_DROP = 0.25
ADAPTIVE_KD_ERROR_BOOST = 0.35
ADAPTIVE_KD_DERIVATIVE_BOOST = 0.55
ADAPTIVE_KI_ERROR_REDUCTION = 0.70
ADAPTIVE_DERIVATIVE_REF = 4.0
INTEGRAL_LIMIT = 1.2
LOW_CONFIDENCE_SPEED_SCALE = 0.65
ERROR_SPEED_REDUCTION = 0.55
DERIVATIVE_SPEED_REDUCTION = 0.35
STRAIGHT_SPEED_BOOST = 0.10
CURVE_SPEED_REDUCTION = 0.45
CURVE_TURN_BOOST = 0.35
MIN_CURVE_SPEED_SCALE = 0.45
NEAR_TRAJECTORY_WEIGHT = 0.65
LOOKAHEAD_TRAJECTORY_WEIGHT = 0.35
LOOKAHEAD_CONFIDENCE_MIN = 0.25
RK4_TRAJECTORY_GAIN = 10.0
RK4_HEADING_GAIN = 8.0
RK4_MAX_DT = 0.05

# Second RK4 layer: tracks curvature trend between the lookahead and far ROIs,
# so a closing-out turn can be anticipated before the near ROI ever straightens.
FAR_CONFIDENCE_MIN = 0.25
RK4_CURVATURE_GAIN = 10.0
RK4_CURVATURE_RATE_GAIN = 8.0
CURVATURE_DAMPING = 0.6

# Smooths the discrete error derivative before it feeds the PID/turn-boost math.
# Raw frame-to-frame derivative of a noisy vision error is the classic cause of
# oscillation ("swinging") on turns -- this trades a little responsiveness for
# a lot less overshoot-correct-overshoot.
DERIVATIVE_SMOOTHING = 0.4

# Persistence of vision. The camera is tilted 25 deg forward, so it sees a turn
# before the wheels reach it. The delay is how long the ground under the near ROI
# takes to reach the wheels. Derived from fused speed when odometry is healthy,
# which is why the estimate has to be smooth -- an earlier version computed it
# from a single raw depth pixel and swung the loop's phase lag 141-450ms frame to
# frame, which oscillates at any gain.
VISION_DELAY_S = 0.12
VISION_DELAY_MIN_S = 0.05
VISION_DELAY_MAX_S = 0.30
VISION_DELAY_FROM_ODOMETRY = True
# Reject speeds this far from the previous estimate as flow outliers.
ODOMETRY_MAX_SPEED_MPS = 1.5

# --- Speed estimation: IMU + visual odometry -------------------------------
# There are no wheel encoders, so absolute speed comes from fusing two sources
# with opposite failure modes:
#   * IMU accel integrates at ~200Hz but drifts within seconds
#   * BEV optical flow is drift-free and absolute but noisy at ~30Hz
# The complementary filter runs on the IMU and lets flow continuously reset its
# drift. ODOMETRY_ALPHA is the IMU's weight per fused update: higher = smoother
# but slower to correct drift.
ODOMETRY_ENABLED = True
ODOMETRY_ALPHA = 0.85
# Flow below this is indistinguishable from sensor noise; treat as stopped.
ODOMETRY_MIN_FLOW_PX = 0.35
# Track points for flow. Fewer is faster; the floor is low-texture so don't
# expect many good corners.
ODOMETRY_MAX_CORNERS = 120
ODOMETRY_QUALITY = 0.02
ODOMETRY_MIN_DISTANCE_PX = 12
# Discard the frame's flow if fewer than this many points survive tracking.
ODOMETRY_MIN_TRACKED = 8
# The D435i IMU is NOT hardware-synced to the frames, so accel samples carry
# their own timestamps and must be integrated on those, not on frame dt.
IMU_ENABLED = True
IMU_ACCEL_FPS = 200
IMU_GYRO_FPS = 200
# Gravity leaks into forward accel whenever the chassis pitches (ramps, speed
# bumps). Above this the accel sample is distrusted and flow carries the estimate.
IMU_MAX_PITCH_DEV_DEG = 12.0

# SPEEDS
SHARP_TURN_SPEED = 0.50
MAX_TURN_SPEED = 0.65
# Negate the turn term on its way to the motors. With the right side no longer
# mirrored in pi_motors, left = f + t already yaws toward a positive error, so
# no extra flip is wanted. This was only ever True to compensate for that
# mirror, which also meant yaw did not depend on the turn term at all.
STEER_INVERT = False
BASE_SPEED = 0.26
MAX_SPEED = 0.42
MIN_SPEED = 0.1
SPIN_SEARCH_SPEED = 0.32
APPROACH_SPEED = 0.24
LINE_LOST_STOP_TIMEOUT_S = 4.0
SLEW_RATE_PER_S = 0.45
# Turn was previously applied unlimited while forward was slew-limited, so a turn
# could jump full-scale in one frame. Higher than SLEW_RATE_PER_S because steering
# must still be responsive.
TURN_SLEW_RATE_PER_S = 2.5
# Below this the BTS7960 just buzzes the gearbox without turning it. Commands
# under the threshold are zeroed rather than scaled up.
MOTOR_MIN_DUTY = 0.06
# A wheel must be commanded past this magnitude in the opposite direction before
# its H-bridge direction bit flips. Without it, a near-zero inner-wheel command
# toggles direction every frame and the driver slams forward/reverse.
MOTOR_DIR_FLIP_HYSTERESIS = 0.05
MANUAL_SPEED = 0.22
MANUAL_KEY_TIMEOUT_S = 0.5

# ROI VALUES
ROI_Y_START_RATIO = 0.67
ROI_Y_END_RATIO = 0.86
ROI_X_START_RATIO = 0.05
ROI_X_END_RATIO = 0.95
LOOKAHEAD_ROI_Y_START_RATIO = 0.42
LOOKAHEAD_ROI_Y_END_RATIO = 0.62
LOOKAHEAD_ROI_X_START_RATIO = 0.05
LOOKAHEAD_ROI_X_END_RATIO = 0.95
FAR_ROI_Y_START_RATIO = 0.24
FAR_ROI_Y_END_RATIO = 0.40
FAR_ROI_X_START_RATIO = 0.05
FAR_ROI_X_END_RATIO = 0.95
WIDE_ROI_Y_START_RATIO = 0.20
WIDE_ROI_X_START_RATIO = 0.0
WIDE_ROI_X_END_RATIO = 1.0

# THRESHOLDS
BLACK_THRESHOLD = 90
MIN_LINE_AREA = 45
PICAMERA2_RGB_TO_BGR = False
GREEN_DIFF_THRESHOLD = 40
RED_DIFF_THRESHOLD = 40
MIN_MARKER_AREA = 40
MARKER_ACTION_DELAY_S = 1.0

# Camera
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
USB_CAMERA_INDEX = 0

# Motor pins (BCM numbering) -- RPi drives the two IBT-2/BTS7960 boards directly.
# Each board takes an LPWM/RPWM pair rather than the L298N ENA+IN1+IN2 triple:
# to drive a side you PWM one input and hold the other at 0.
#
# All chosen pins are in BCM 9-27, which default to pull-DOWN, so the bridges
# read LOW through boot before any code runs. Pins 0-8 default HIGH and must
# never be used here. SPI0 (7-11), I2C1 (2,3) and the UART console (14,15) are
# left free.
LEFT_LPWM = 13
LEFT_RPWM = 12
RIGHT_LPWM = 19
RIGHT_RPWM = 18

# R_EN/L_EN on each board. Tie both of a board's enables to one Pi pin. Set to
# None if you have strapped them to +5V in hardware instead.
LEFT_EN = 20
RIGHT_EN = 21

# Per-side polarity. The IBT-2 boards are symmetric, so whether a side runs
# backwards depends only on how its motor leads are landed -- that is wiring
# data, which is why it lives beside the pin numbers instead of in the driver.
# Re-verify with _scratch_sides.py after any rewiring.
LEFT_MOTOR_INVERT = False
RIGHT_MOTOR_INVERT = False

# pigpio's DMA-timed PWM only offers 18 frequencies, and which ones depend on
# pigpiod's sample rate (default 5us). 10 kHz is NOT selectable at 5us -- it
# clamps to 8000, whose real range is 1e6/(5*8000) = 25 duty steps, i.e. 4% per
# step. That is coarser than MOTOR_MIN_DUTY, so the minimum-duty threshold
# landed between steps. 2 kHz gives 100 steps (1%) and is still inaudible-ish.
#
# Hardware PWM is deliberately NOT used: only GPIO 12/13/18/19 have it, they map
# to just two channels (12/18 -> 0, 13/19 -> 1), and a channel's duty is shared
# by every GPIO on it -- so driving all four collapsed the two sides into one.
PWM_FREQUENCY_HZ = 2000

# Cut the motors if the control loop stops feeding commands. With no
# microcontroller latching state, a crashed process would otherwise leave the
# last duty applied indefinitely.
MOTOR_COMMAND_TIMEOUT_S = 0.3

# Indicator pins. Moved off BCM 5/6/13: 13 collided with LEFT_LPWM, and 5/6 sit
# in the pull-UP bank so an LED would glow through boot.
STATUS_LED_PIN = 26
RED_LED_PIN = 22
GREEN_LED_PIN = 23
BUZZER_PIN = 24

# Acceleration buzzer
ACCEL_BUZZER_ENABLED = True
ACCEL_BUZZER_FREQ_HZ = 220.0
ACCEL_BUZZER_DUTY = 30.0
ACCEL_SPEED_DELTA_THRESHOLD = 0.01

# Intersection detection
INTERSECTION_MIN_CONTOURS = 2
INTERSECTION_MIN_AREA = 40

# Temporal buffer / backtracking
MAX_WAYPOINTS = 50
BACKTRACK_SPEED = 0.18
ROTATE_SPEED = 0.38
BACKTRACK_SEARCH_TIMEOUT_S = 8.0
INTERSECTION_COOLDOWN_S = 2.0
INTERSECTION_MEMORY_MAX_AGE_S = 60.0
ROTATE_SETTLE_TIME_S = 0.25

# Debug
# Two imshow windows per frame on the Pi inflates and destabilises loop dt, which
# directly scales the PID derivative term. Turn off when tuning motion.
SHOW_DEBUG_VIEW = False
DEBUG_VIEW_EVERY_N_FRAMES = 2
LOOP_LOG_INTERVAL_S = 0.20
ROI_VIEW_SCALE = 3
ARROW_MAX_DEFLECTION_DEG = 65.0

# RealSense & Perspective Configs
CAMERA_TILT_ANGLE_DEG = 25.0
CAMERA_MOUNT_HEIGHT_MM = 60.0

WARP_MATRIX = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
]

# Bird's-eye ground patch. Rectifying to a known metric rectangle makes ROI rows
# true distances instead of image ratios, and makes optical flow metrically
# scaled for free (see BEV_MM_PER_PX) -- which is how a monocular camera gets an
# absolute speed without encoders.
#
# Distances are from the FRONT AXLE, since the drive wheels are at the front.
# The near edge starts at 60mm because the camera cannot see ground closer than
# ~58mm at this mount height/tilt.
BEV_ENABLED = True
BEV_NEAR_MM = 60.0
BEV_FAR_MM = 460.0
BEV_WIDTH_MM = 400.0
BEV_OUTPUT_PX = 400
BEV_MM_PER_PX = BEV_WIDTH_MM / BEV_OUTPUT_PX
# Camera optical centre offset from the robot centreline, +ve = mounted right.
BEV_CAMERA_X_OFFSET_MM = 0.0
# Forward gap from the front axle to the camera lens. The near ROI images ground
# only ~71-96mm ahead of the LENS, so if the lens sits ahead of the axle that
# patch is nearly under the wheels already.
BEV_CAMERA_AXLE_OFFSET_MM = 0.0
# Saved by calibrate_bev.py; overrides the analytic derivation when present.
BEV_CALIBRATION_FILE = "bev_calibration.json"

# Obstacle Detection
GROUND_DEPTH_TOLERANCE_MM = 40.0 # Anything closer by this amount is an obstacle
# Disabled: at CAMERA_MOUNT_HEIGHT_MM=60 the near ROI images ground only 71-96mm
# ahead, inside the D435i minimum-depth blind zone (~105mm+), so depth there is 0
# or garbage. The mask was flagging every ground pixel as an obstacle, erasing the
# line and flapping a hard +/-0.6 steering offset at frame rate. Re-enabling needs
# a higher camera mount, not a bigger tolerance.
OBSTACLE_DETECTION_ENABLED = False
# Competition obstacles are >=15cm high (RescueLine rules 3.5.4) and have NO
# specified colour -- red marks the goal tile and the dead-victim point instead,
# so colour must not be used as the obstacle cue. A 150mm object is ~2.5x the
# camera height: a large depth step, well outside the near-field blind zone.
OBSTACLE_MIN_HEIGHT_MM = 100.0
OBSTACLE_MIN_AREA_PX = 400

# --- Obstacle band detection + go-around ----------------------------------
# Separate from the per-pixel mask above, which stays off. Because the lens sits
# only 60mm up and every FOV ray points downward, anything taller than the mount
# intercepts the WHOLE image column: a 15cm obstacle is a full-height vertical
# BAND, not a blob. That is a much stronger cue than a per-pixel depth step, and
# it needs no colour -- which matters because red marks the goal tile and the
# dead-victim point, not obstacles.
OBSTACLE_BAND_ENABLED = True

# A downward ray reaches the floor before an obstacle whenever the floor is
# nearer, so an obstacle does NOT fill its columns -- it is anchored at the top
# of the frame and grows downward as range closes (~7% of frame height at 500mm,
# ~42% at 150mm). The cue is a wide run of columns reading nearer than the
# expected GROUND depth, using GROUND_DEPTH_TOLERANCE_MM as the margin.
#
# A band must be this wide (as a fraction of frame width) to count. Rejects
# debris and single-column depth noise.
OBSTACLE_BAND_MIN_WIDTH_RATIO = 0.12
# Ignore anything farther than this; far bands are walls and course furniture.
OBSTACLE_BAND_MAX_RANGE_MM = 500.0
# Consecutive frames a band must persist before it is believed.
OBSTACLE_BAND_MIN_FRAMES = 3

# The D435 cannot measure closer than ~105-280mm depending on preset, so depth
# goes to zero exactly at the 80mm standoff we want. The dropout IS the trigger:
# once a tracked band's depth vanishes we are at the near limit. Distance is
# therefore imprecise and preset-dependent -- a deliberate trade for not needing
# to dead-reckon through the blind zone.
OBSTACLE_TRIGGER_ON_DROPOUT = True
# A band must have been this close before its dropout is trusted, so a band
# lost to noise at long range does not fire the manoeuvre.
OBSTACLE_DROPOUT_MAX_RANGE_MM = 320.0
# Consecutive all-invalid frames required. Depth flickers; one frame is noise.
OBSTACLE_DROPOUT_FRAMES = 2

ROBOT_WIDTH_MM = 200.0
# Obstacles occupy at most one tile (300mm). Clearing a worst-case one needs
# obstacle half-width + robot half-width + margin. This exceeds the 150mm of
# on-tile room either side of the line, so the robot briefly leaves the tile --
# the rules score navigating AROUND an obstacle, not staying on-tile.
OBSTACLE_CLEARANCE_MARGIN_MM = 40.0
OBSTACLE_LATERAL_OFFSET_MM = 150.0 + ROBOT_WIDTH_MM / 2.0 + OBSTACLE_CLEARANCE_MARGIN_MM
# Forward run past the obstacle before cutting back toward the line. One tile
# plus the robot's own length worth of slack.
OBSTACLE_PASS_FORWARD_MM = 380.0
# Speeds for the manoeuvre. Deliberately below BASE_SPEED: the legs are
# dead-reckoned, and odometry error grows with speed.
OBSTACLE_MANEUVER_SPEED = 0.20
OBSTACLE_MANEUVER_TURN = 0.55
# Give up on a leg after this long even if the odometry target is never met,
# so a stalled wheel or bad flow estimate cannot hang the manoeuvre forever.
OBSTACLE_LEG_TIMEOUT_S = 6.0
# After the pass, sweep this far looking for the line before declaring failure
# and retrying on the other side.
OBSTACLE_REACQUIRE_MM = 260.0
