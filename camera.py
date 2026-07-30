import numpy as np
import cv2

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

from config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, USB_CAMERA_INDEX,
    IMU_ENABLED, IMU_ACCEL_FPS, IMU_GYRO_FPS,
)

def _open_usb_camera():
    import sys
    backend = cv2.CAP_DSHOW if sys.platform.startswith('win') else cv2.CAP_ANY
    
    cap = cv2.VideoCapture(USB_CAMERA_INDEX, backend)
    if not cap.isOpened():
        # Fallback to default if DSHOW fails
        cap = cv2.VideoCapture(USB_CAMERA_INDEX)
        if not cap.isOpened():
            return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # FIX JITTER: Disable OpenCV frame buffering

    # Warm up the camera
    for _ in range(5):
        cap.read()

    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"USB camera opened on index {USB_CAMERA_INDEX} at {actual_w}x{actual_h}.")
    print("No depth stream: obstacle detection is disabled, line following only.")
    return cap

def profile_device_usb():
    """USB spec the RealSense is attached at, e.g. 2.1 or 3.2. None if unknown."""
    devices = rs.context().query_devices()
    if len(devices) == 0:
        return None
    desc = devices[0].get_info(rs.camera_info.usb_type_descriptor)
    try:
        return float(desc)
    except (TypeError, ValueError):
        return None


def open_camera():
    if rs is not None:
        pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, CAMERA_FPS)
        config.enable_stream(rs.stream.depth, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.z16, CAMERA_FPS)

        # The D435 (non-i) has no IMU, so a motion stream request hard-fails the
        # whole pipeline. Try with motion, then retry without.
        #
        # A USB-2 link is the nastier case: depth + color + IMU exceeds its
        # bandwidth, but the pipeline still STARTS. Frames then never arrive and
        # wait_for_frames times out far from here, so drop the IMU up front
        # rather than letting it look like a stream-rate bug.
        want_imu = IMU_ENABLED
        if want_imu:
            usb = None
            try:
                usb = profile_device_usb()
            except Exception:
                usb = None
            if usb is not None and usb < 3.0:
                print(f"USB {usb} link: too little bandwidth for depth+color+IMU. "
                      "Disabling IMU -- move the camera to the blue USB-3 port "
                      "to get odometry back.")
                want_imu = False
        if want_imu:
            try:
                config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, IMU_ACCEL_FPS)
                config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, IMU_GYRO_FPS)
            except RuntimeError as e:
                print(f"IMU streams unavailable ({e}); continuing without.")
                want_imu = False

        profile = None
        try:
            profile = pipeline.start(config)
        except RuntimeError as e:
            if want_imu:
                print(f"Start failed with IMU enabled ({e}); retrying without IMU.")
                want_imu = False
                config = rs.config()
                config.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, CAMERA_FPS)
                config.enable_stream(rs.stream.depth, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.z16, CAMERA_FPS)
                try:
                    profile = pipeline.start(config)
                except RuntimeError as e2:
                    e = e2

        if profile is None:
            print(f"Requested configuration ({CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS} FPS) not supported: {e}")
            print("Falling back to default camera configuration.")
            want_imu = False
            try:
                profile = pipeline.start()
            except RuntimeError as e2:
                print(f"RealSense unavailable: {e2}")
                profile = None

        if profile is not None:
            align = rs.align(rs.stream.color)
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            print(f"RealSense started. Depth scale: {depth_scale}. Intrinsics: {intrinsics.width}x{intrinsics.height}")
            print(f"IMU: {'enabled' if want_imu else 'disabled'}")
            return "realsense", (pipeline, align, depth_scale, intrinsics, want_imu)
    else:
        print("pyrealsense2 not available.")

    print("Trying USB camera fallback...")
    cap = _open_usb_camera()
    if cap is not None:
        return "usb", cap

    print("No USB camera found, falling back to mock")
    return "mock", None

def read_frame(camera_kind, camera):
    if camera_kind == "mock":
        # Create a blank mock frame
        color_image = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        depth_image = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH), dtype=np.uint16)
        return color_image, depth_image
    elif camera_kind == "usb":
        ok, color_image = camera.read()
        if not ok or color_image is None:
            return None, None
        return color_image, None
    elif camera_kind == "realsense":
        pipeline, align = camera[0], camera[1]
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        return color_image, depth_image
    return None, None


def read_frame_with_motion(camera_kind, camera):
    """Frame plus any IMU samples that arrived with it.

    Returns (color, depth, motion) where motion is a list of
    ('accel'|'gyro', (x, y, z), timestamp_s). The D435i IMU is not
    hardware-synced to the frames, so each sample carries its own device
    timestamp and must be integrated on that, not on frame dt.
    """
    if camera_kind != "realsense" or not isinstance(camera, tuple):
        color, depth = read_frame(camera_kind, camera)
        return color, depth, []

    pipeline, align = camera[0], camera[1]
    imu_on = camera[4] if len(camera) > 4 else False

    frames = pipeline.wait_for_frames()

    motion = []
    if imu_on:
        for f in frames:
            m = f.as_motion_frame()
            if not m:
                continue
            d = m.get_motion_data()
            prof = m.get_profile().stream_type()
            kind = "accel" if prof == rs.stream.accel else "gyro"
            # Device timestamps are milliseconds.
            motion.append((kind, (d.x, d.y, d.z), m.get_timestamp() / 1000.0))
        motion.sort(key=lambda s: s[2])

    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None, None, motion

    return (
        np.asanyarray(color_frame.get_data()),
        np.asanyarray(depth_frame.get_data()),
        motion,
    )


def has_imu(camera):
    return isinstance(camera, tuple) and len(camera) > 4 and bool(camera[4])


def get_intrinsics(camera):
    if not isinstance(camera, tuple):
        return None
    return camera[3]

def get_depth_scale(camera):
    if not isinstance(camera, tuple):
        return 0.001 # standard 1mm default
    return camera[2]
