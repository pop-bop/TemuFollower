import cv2
import threading
import time

from config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, PICAMERA2_RGB_TO_BGR

class CameraError(RuntimeError):
    pass


class ThreadedCamera:
    def __init__(self, camera_kind, camera):
        self.camera_kind = camera_kind
        self.camera = camera
        self.frame = None
        self.running = True
        self.error = None
        self.lock = threading.Lock()
        self._capture()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _capture(self):
        if self.camera_kind == "picamera2":
            rgb = self.camera.capture_array()
            if PICAMERA2_RGB_TO_BGR:
                f = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            else:
                f = rgb
        else:
            ok, f = self.camera.read()
            if not ok:
                f = None
        with self.lock:
            self.frame = f

    def _update(self):
        while self.running:
            try:
                self._capture()
            except Exception as exc:
                with self.lock:
                    self.error = exc
                self.running = False
                break
            time.sleep(0.001)

    def read(self):
        with self.lock:
            if self.error is not None:
                raise CameraError(f"camera capture thread died: {self.error}") from self.error
            if self.frame is not None:
                return self.frame.copy()
            return None

    def stop(self):
        self.running = False
        self.thread.join()
        if self.camera_kind == "picamera2":
            self.camera.stop()
        else:
            self.camera.release()

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
        print("Using Picamera2 (Threaded)")
        return "picamera2", ThreadedCamera("picamera2", picam2)
    except Exception as exc:
        print(f"Picamera2 unavailable, falling back to cv2.VideoCapture: {exc}")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("Using cv2.VideoCapture (Threaded)")
        return "cv2", ThreadedCamera("cv2", cap)

def read_frame(camera_kind, camera):
    return camera.read()
