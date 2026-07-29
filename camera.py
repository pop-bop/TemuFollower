import numpy as np
import cv2

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

from config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, USB_CAMERA_INDEX

def _open_usb_camera():
    cap = cv2.VideoCapture(USB_CAMERA_INDEX)
    if not cap.isOpened():
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"USB camera opened on index {USB_CAMERA_INDEX} at {actual_w}x{actual_h}.")
    print("No depth stream: obstacle detection is disabled, line following only.")
    return cap

def open_camera():
    if rs is not None:
        pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, CAMERA_FPS)
        config.enable_stream(rs.stream.depth, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.z16, CAMERA_FPS)

        try:
            profile = pipeline.start(config)
        except RuntimeError as e:
            print(f"Requested configuration ({CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS} FPS) not supported: {e}")
            print("Falling back to default camera configuration.")
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
            return "realsense", (pipeline, align, depth_scale, intrinsics)
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
        pipeline, align, _, _ = camera
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

def get_intrinsics(camera):
    if not isinstance(camera, tuple):
        return None
    _, _, _, intrinsics = camera
    return intrinsics

def get_depth_scale(camera):
    if not isinstance(camera, tuple):
        return 0.001 # standard 1mm default
    _, _, depth_scale, _ = camera
    return depth_scale
