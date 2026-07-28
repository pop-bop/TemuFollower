import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

from config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS

def open_camera():
    if rs is None:
        print("pyrealsense2 not available, falling back to mock")
        return "mock", None

    pipeline = rs.pipeline()
    config = rs.config()
    
    config.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, CAMERA_FPS)
    config.enable_stream(rs.stream.depth, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.z16, CAMERA_FPS)
    
    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    
    print(f"RealSense started. Depth scale: {depth_scale}. Intrinsics: {intrinsics.width}x{intrinsics.height}")
    return "realsense", (pipeline, align, depth_scale, intrinsics)

def read_frame(camera_kind, camera):
    if camera_kind == "mock":
        # Create a blank mock frame
        color_image = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        depth_image = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH), dtype=np.uint16)
        return color_image, depth_image
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
    if camera is None:
        return None
    _, _, _, intrinsics = camera
    return intrinsics

def get_depth_scale(camera):
    if camera is None:
        return 0.001 # standard 1mm default
    _, _, depth_scale, _ = camera
    return depth_scale
