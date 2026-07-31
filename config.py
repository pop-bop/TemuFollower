# PID
# old_working ran KP=1.10 at 120fps and tracked cleanly. The detune to 0.10 was
# an attempt to stop the hunting by starving the loop of authority, but that
# treats the symptom: with the P term this small the turn command is dominated
# by the SHARP_TURN_SPEED blend below, which injects turn by the SIGN of the
# error alone. That is bang-bang steering -- it cannot settle, and it hunts
# regardless of how low KP goes. The real cause was phase lag (see CAMERA_FPS);
# with the frame period halved, the loop can carry real proportional gain again.
# Kept a little under old_working's 1.10 because the loop is still slower than
# 120fps.
KP = 0.85
KI = 0.02
KD = 0.14
TURN_LIMIT = 0.80
CENTER_DEADZONE = 0.12
# Squaring the error to soften small corrections: also part of the above
# compensation, and it silently halves the effective KP everywhere the line is
# only slightly off-centre, which is most of a lap.
NONLINEAR_ERROR_MAPPING = False
# Deadband compensation is OFF. old_working had none and drove fine: the
# BTS7960's stiction is already handled at the driver by MOTOR_MIN_DUTY, and a
# flat additive step here is a discontinuity in the middle of the control
# range, which is its own oscillation source.
TURN_DEADBAND = 0.0
ADAPTIVE_PID_ENABLED = True
# Was 0.55, which pushed KP from 1.10 to ~1.45 exactly when the error was
# largest -- on top of the SHARP_TURN_SPEED blend, which is already injecting
# turn proportional to that same error. Two error-proportional boosts stacked on
# one loop drove 73% turn-sign flips over 12 consecutive FOLLOW frames on the
# 2026-07-30 runs. The blend is the one that has to stay (it is how the robot
# gets around corners), so this one gives way.
ADAPTIVE_KP_ERROR_BOOST = 0.0
ADAPTIVE_KP_CONFIDENCE_DROP = 0.50
ADAPTIVE_KD_ERROR_BOOST = 1.0
ADAPTIVE_KD_DERIVATIVE_BOOST = 1.0
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
LOOKAHEAD_CONFIDENCE_MIN = 0.25
RK4_TRAJECTORY_GAIN = 10.0
RK4_HEADING_GAIN = 8.0
RK4_MAX_DT = 0.05

# Second RK4 layer: tracks curvature trend between the lookahead and far ROIs,
# so a closing-out turn can be anticipated before the near ROI ever straightens.
#
# CURVATURE_DAMPING = 0.0 disables its contribution. old_working ran ONE RK4 layer
# and had no far ROI at all; stacking a second predictor on a loop that lost 4x its
# bandwidth adds phase lag exactly where the oscillation shows up. Re-enable (try
# 0.3) only once the loop rate is back and the single-layer behaviour is stable.
FAR_CONFIDENCE_MIN = 0.25
RK4_CURVATURE_GAIN = 10.0
RK4_CURVATURE_RATE_GAIN = 8.0
CURVATURE_DAMPING = 0.0

# Smooths the discrete error derivative before it feeds the PID/turn-boost math.
# Weight of the NEW raw sample per frame, so lower = more smoothing = more lag.
# old_working used the RAW derivative (no smoothing at all) and was stable,
# because at 120fps a single noisy frame is 8ms of influence. Every value below
# 1.0 buys noise rejection with phase lag, and phase lag is what was making
# this robot hunt. At 60fps a light touch is affordable; 0.7 is ~1.4 frames of
# memory, about 12ms, versus the 68ms that 0.4 cost at 22fps.
DERIVATIVE_SMOOTHING = 0.70
# Hard bound on the error derivative fed to the D term, in error-units/second.
# The observed failure: with the robot yawing, the line sweeps across the image
# at up to ~4 err/s, the D term saturates (KD*4 = 0.9, beyond TURN_LIMIT) and
# steering follows the sweep rather than the line -- frame captured at
# error=+0.035 with turn=-0.800. A genuine tracking error changes well under
# 1.5 err/s at these speeds, so clamp there; yaw-induced sweeps get cut, real
# corrections pass through.
DERIVATIVE_CLAMP = 1.5

# Persistence of vision. The camera looks down 22.5 deg from horizontal, so the
# near ROI is ground ~90mm AHEAD of the wheels: the robot must steer for the
# line it SAW vision_delay ago, which is what is under the wheels now. At the
# observed 0.15-0.3 m/s that 90mm takes 0.3-0.6s; 0.35 splits the band. Fixed
# value -- the odometry-derived variant needs the IMU, which is off on USB-2.
VISION_DELAY_S = 0.0
VISION_DELAY_MIN_S = 0.05
VISION_DELAY_MAX_S = 0.60
VISION_DELAY_FROM_ODOMETRY = False
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
ODOMETRY_ENABLED = False
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
IMU_ENABLED = False
IMU_ACCEL_FPS = 200
IMU_GYRO_FPS = 200
# Gravity leaks into forward accel whenever the chassis pitches (ramps, speed
# bumps). Above this the accel sample is distrusted and flow carries the estimate.
IMU_MAX_PITCH_DEV_DEG = 12.0

# SPEEDS
# Turn injected purely by the SIGN of the error once |err| exceeds
# CENTER_DEADZONE, blended in proportionally. old_working used 0.50; this
# branch raised it to 0.70 while KP was cut to 0.10, which meant the blend
# supplied 78-100% of the turn command over most of the error range -- the
# steering became a function of sign(error) alone, with no proportional
# component left to settle it. Back to old_working's value now that KP carries
# real authority again.
SHARP_TURN_SPEED = 0.50
MAX_TURN_SPEED = 0.65
# Global governor on forward speed, applied last on the way to the motors. 1.0 is
# a no-op; lower it for floor testing so a runaway cannot build momentum into a
# wall. Set via the SPEED_CAP env var so a test run never edits this file.
SPEED_CAP_SCALE = float(__import__("os").environ.get("SPEED_CAP", "1.0"))
# Negate the turn term on its way to the motors. False: both sides are inverted
# in the driver (see LEFT/RIGHT_MOTOR_INVERT), and negating BOTH sides is
# symmetric -- it reverses forward but leaves the left-right difference, i.e. the
# yaw, pointing the same way. So the wiring fix needs no steering compensation.
# Verified against the error sign: err>0 (line right of centre) must yaw right.
STEER_INVERT = False
# old_working ran these at 120fps where the loop reacted in ~8ms. At the current
# ~33ms/frame the robot covers 4x the distance per control decision, and the near
# ROI only sees ~71-96mm ahead -- at 0.26 base the corner is underneath the robot
# before the turn slew winds up (measured: clean straight-line FOLLOW streaks that
# break exactly when turn rails at a corner). Scale speed down with loop rate.
# The old pivot failure at low BASE_SPEED does not apply anymore: the MIN_SPEED
# floor is unconditional again, so forward never collapses to zero.
BASE_SPEED = 0.25
MAX_SPEED = 0.45
MIN_SPEED = 0.1
SPIN_SEARCH_SPEED = 0.32
APPROACH_SPEED = 0.24
# Bounds on what the WIDE ROI may treat as a line worth driving toward. Measured
# on the real mat: a genuine line reads ~33600px there. Below the floor it is a
# speck or a shadow edge; above the ceiling it is the mat edge or a seam running
# across the frame. Without these APPROACH accepted anything from 180px up to
# the 147456px full-ROI guard and drove at it.
WIDE_MIN_LINE_AREA = 4000
WIDE_MAX_LINE_AREA = 90000
# APPROACH runs when the near ROI has already lost the line, i.e. on the robot's
# least reliable evidence, so it steers gently rather than at the FOLLOW rail.
APPROACH_TURN_LIMIT = 0.45
LINE_LOST_STOP_TIMEOUT_S = 4.0
# How many consecutive near-ROI misses to coast through before believing the
# line is really gone. The near ROI sees a ~25mm-deep strip of ground, so at
# BASE_SPEED a single miss is usually the line slipping out of it mid-correction.
# 3 frames is 100ms at 30fps -- long enough to ride out that, short enough that a
# genuinely lost line still reaches SPIN_SEARCH promptly.
LINE_LOST_GRACE_FRAMES = 3
SLEW_RATE_PER_S = 0.45
# Turn slew rate. This was the single largest lag term: at 5.0/s the turn needs
# 140ms to reach full authority, on a loop whose near ROI only shows ~100ms of
# ground ahead. The correction was still ramping up while the ground it was
# meant for went under the wheels, so it kept arriving late and in the wrong
# direction -- textbook phase-lag oscillation.
#
# old_working had NO turn slew at all (main.py:205 applied a plain clamp) and
# was stable, because at 120fps one frame of full-authority turn is only 8ms of
# yaw. The slew was introduced here to tame the hunting, which made it worse.
#
# 25.0/s at 60fps allows 0.42 change per frame: fast enough that the turn is
# essentially applied on the frame it is computed (~28ms to full authority),
# while still preventing a single garbage frame from slamming rail to rail.
TURN_SLEW_RATE_PER_S = 25.0
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
# Measured inside the running loop on a static scene: frame.min() wanders 9-47
# as the RealSense re-exposes. A global cut keeps pixels strictly BELOW it, so
# at 45 the mask went EXACTLY empty on any frame whose darkest pixel reached 45
# -- 147 of 207 frames reported LINE LOST while parked on the line, which is the
# FOLLOW/SPIN_SEARCH flapping. 90 is old_working's value and leaves ~45 counts
# of headroom above the darkest observed pixel.
BLACK_THRESHOLD = 90
# Pixel areas scale with resolution. 45 was tuned at 320x240, so at 640x480 the
# same real line covers ~4x the pixels: the gate stopped rejecting anything, and
# line_confidence (area / MIN_LINE_AREA*8) pinned to 1.0 on every frame. That
# silently disabled LOW_CONFIDENCE_SPEED_SCALE, ADAPTIVE_KP_CONFIDENCE_DROP and
# both confidence gates -- measured 9700px on a plain straight line.
MIN_LINE_AREA = 180
# Area at which line_confidence reaches 1.0. A plain straight line measures
# ~9700px in the near ROI at 640x480, so full confidence sits a little under
# that and a half-visible line scores ~0.5 instead of saturating.
LINE_CONFIDENCE_FULL_AREA = 8000
GREEN_DIFF_THRESHOLD = 40
RED_DIFF_THRESHOLD = 40
MIN_MARKER_AREA = 800
MARKER_ACTION_DELAY_S = 1.0

# --- Green intersection markers (RescueLine 3.6) ---------------------------
# 25x25mm green squares sit JUST BEFORE an intersection and name the exit:
# marker left of the line = turn left, right = turn right, one on EACH side =
# dead end, turn around. No marker = straight on. The near ROI is the "just
# before the wheels" view, so the manoeuvre fires when the latched marker
# slips out of it: advance to put the wheels over the intersection centre,
# then pivot until the exit line is under the near ROI again.
GREEN_ADVANCE_S = 0.7           # marker-under-ROI to intersection-centre at APPROACH_SPEED
GREEN_PIVOT_TURN = 0.55
GREEN_PIVOT_MIN_S = 0.5         # leave the current line before accepting a new one
GREEN_PIVOT_MIN_S_UTURN = 1.4   # a U-turn must swing past the first 90 degrees
GREEN_PIVOT_TIMEOUT_S = 3.5

# --- Green marker decision graph (markers.py) ------------------------------
# Rows the marker may occupy, as a fraction of frame height. This IS rule
# 3.6.6 ("markers are placed just before the intersection"): only ground the
# robot is about to reach can vote. A marker beyond the intersection belongs
# to a robot approaching from another branch.
MARKER_BAND_Y_START = 0.55
MARKER_BAND_Y_END = 0.98
# How far from the line's fitted centre a blob may sit and still be its
# marker. A 25mm marker abuts a 25mm line, so a couple of line widths. Beyond
# this it is evacuation-zone green or off-tile clutter.
MARKER_MAX_LINE_DIST_PX = 90
# Consecutive frames the same verdict must repeat before it is acted on. One
# frame of green is a reflection; a real marker persists as the robot closes.
MARKER_CONFIRM_FRAMES = 3

# --- Follow-the-gap depth avoidance (see depth_nav.py) ----------------------
# Steers around the obstacle on live depth while it is still 300-500mm out,
# instead of waiting for it to enter the blind zone and then dead-reckoning.
GAP_NAV_ENABLED = True
# Columns masked either side of the closest return before choosing a gap --
# the "safety bubble". Must cover the robot's half-width at the obstacle's
# range: ~100mm at 350mm subtends roughly 70px at this FOV.
GAP_BUBBLE_COLS = 70
# A gap narrower than the robot is not a gap. 200mm robot at ~350mm range.
GAP_MIN_WIDTH_COLS = 140
# Rows in a column that must read nearer-than-floor before it counts as
# occupied. Rejects single-row depth noise.
GAP_COLUMN_MIN_HITS = 6
# Above this fraction of invalid depth a column is unmeasurable, not clear.
# This is the near-obstacle case: too close to return depth at all.
GAP_UNKNOWN_MAX_FRAC = 0.85
# Steering authority for the gap controller, and the range at which it starts
# blending in over the line follower.
GAP_STEER_GAIN = 0.55
GAP_ENGAGE_RANGE_MM = 450.0
# Below this the obstacle fills the view; commit to the timed box instead.
GAP_COMMIT_RANGE_MM = 200.0

# Camera
# THE OSCILLATION FIX. old_working ran 320x240@120fps and was stable; this
# branch moved to 640x480@30 and started hunting. The link is USB-2 (the D435i
# reports usb_type_descriptor 2.1), which sustains roughly 30 MB/s. Colour plus
# depth at 640x480x30 asks for ~37 MB/s, so the pipeline never reached 30fps --
# measured 45ms/frame, i.e. 22fps.
#
# That frame period is the base term in the loop's phase lag, and the lag is
# what oscillates: at 22fps the total control lag was ~254ms while the near ROI
# only shows ~100ms of ground ahead at BASE_SPEED. Every correction landed on
# ground the robot had already crossed, so it corrected the wrong way and
# reversed on the next frame. old_working's ratio was 8.3ms lag against 24.3ms
# horizon -- comfortably inside its own sight line.
#
# 424x240 is a native D435 colour mode (not a crop), so 60fps fits in ~24 MB/s
# with depth. Halving the frame period roughly halves the dominant lag term.
# The ROI ratios are fractions of frame height, so they follow automatically.
CAMERA_WIDTH = 424
CAMERA_HEIGHT = 240
CAMERA_FPS = 60
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
#
# Both True: observed on the track 2026-07-30 -- a steady forward=+0.24 command
# with turn~0 drove the robot BACKWARDS, so both sets of motor leads are landed
# reversed. This is the wiring fact; STEER_INVERT then compensates for the yaw
# sign that inverting both sides flips (see there).
LEFT_MOTOR_INVERT = True
RIGHT_MOTOR_INVERT = True

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
# Also a 320x240-era area: at 640x480 a 40px blob is a speck, so a branch could
# be declared on noise.
INTERSECTION_MIN_AREA = 160

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
# Measured mount: 22.5 degrees below horizontal. This feeds
# get_expected_depth_map, so an error here biases the whole expected-floor
# model and therefore every obstacle decision -- 25.0 was making the predicted
# floor nearer than it really is at every row.
CAMERA_TILT_ANGLE_DEG = 22.5
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
# Disabled while the camera is stuck on a USB-2 link. BEV exists only to give
# optical flow a metric scale for the speed estimate, and the IMU it fuses with is
# force-disabled on USB-2 (see camera.py), so the whole chain costs a 400x400
# warpPerspective plus calcOpticalFlowPyrLK every frame and returns an estimate
# nothing can trust. Turning it off cut measured loop time from ~46ms to ~33ms.
# Re-enable together with the USB-3 move.
BEV_ENABLED = False
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

# --- Obstacle band detection + go-around ----------------------------------
# Separate from the per-pixel mask above, which stays off. Because the lens sits
# only 60mm up and every FOV ray points downward, anything taller than the mount
# intercepts the WHOLE image column: a 15cm obstacle is a full-height vertical
# BAND, not a blob. That is a much stronger cue than a per-pixel depth step, and
# it needs no colour -- which matters because red marks the goal tile and the
# dead-victim point, not obstacles.
# Off while the line-following regression is being chased: OBSTACLE_TRIGGER_ON_DROPOUT
# is False below, so the manoeuvre can never actually fire -- but leaving this True
# still forces the depth stream on, which costs an rs.align over every 640x480 frame
# plus a per-column Python scan in detect_obstacle_band. That is pure loop latency
# in the steering path for a feature that cannot trigger. Re-enable together with
# OBSTACLE_TRIGGER_ON_DROPOUT once the line following is solid again.
OBSTACLE_BAND_ENABLED = True

# Rows below this fraction of the frame image near floor inside the D435i
# blind zone, where USB-2 depth reads garbage non-zero values that beat the
# expected-floor map and paint phantom bands. An upright obstacle is anchored
# at the TOP of the frame (see below), so ignoring the bottom costs nothing.
# This replaces the hand-edit found live on the Pi that zeroed the same rows.
OBSTACLE_BAND_IGNORE_BELOW_FRAC = 0.40

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

# Speeds for the manoeuvre. Deliberately below BASE_SPEED: the legs are
# open-loop, and error grows with speed.
OBSTACLE_MANEUVER_SPEED = 0.20
OBSTACLE_MANEUVER_TURN = 0.65
# Timed "box" legs. There are no encoders and odometry is off (IMU disabled on
# the USB-2 link), so the mm-based legs of the arc go-around had NOTHING to
# measure against: obstacle_leg_done fell through to its 6s timeout on every
# leg, producing a 6-second blind arc. A skid-steer pivots in place cleanly,
# so the manoeuvre is a timed box instead: pivot 90, clear, pivot back,
# pass, pivot toward the line, cross, re-align. Durations carried over from
# the values hand-tuned live on the Pi on 2026-07-31.
OBSTACLE_PIVOT_90_S = 0.8
OBSTACLE_CLEAR_S = 1.2
OBSTACLE_PASS_S = 2.5
# Give up on the return crossing after this long and fall back to searching.
OBSTACLE_RETURN_TIMEOUT_S = 3.0
