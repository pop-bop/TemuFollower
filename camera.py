import threading

import cv2

from config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, PICAMERA2_RGB_TO_BGR


def open_camera():
    cv2.setUseOptimized(True)
    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        video_config = {
            "main": {"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"},
            "controls": {"FrameRate": CAMERA_FPS},
            "buffer_count": 2,
        }
        try:
            config = picam2.create_video_configuration(**video_config, queue=False)
        except TypeError:
            config = picam2.create_video_configuration(**video_config)
        picam2.configure(config)
        picam2.start()
        print("Using Picamera2")
        return "picamera2", picam2
    except Exception as exc:
        print(f"Picamera2 unavailable, falling back to cv2.VideoCapture: {exc}")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("Using cv2.VideoCapture")
        return "cv2", cap


def read_frame(camera_kind, camera):
    if camera_kind == "picamera2":
        rgb = camera.capture_array()
        if PICAMERA2_RGB_TO_BGR:
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return rgb
    else:
        ok, frame = camera.read()
        return frame if ok else None


class FrameGrabber:
    """Reads frames on a background thread so the control loop never blocks
    on camera I/O -- it just grabs whatever the latest frame is."""

    def __init__(self, camera_kind, camera):
        self.camera_kind = camera_kind
        self.camera = camera
        self._lock = threading.Lock()
        self._frame = None
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            frame = read_frame(self.camera_kind, self.camera)
            if frame is not None:
                with self._lock:
                    self._frame = frame

    def get_latest(self):
        with self._lock:
            return self._frame

    def stop(self):
        self._running = False
        self._thread.join(timeout=1.0)
