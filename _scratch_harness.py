"""Harness: run the real main loop, watch the camera live in a browser, stop safely.

Live view is an MJPEG stream rather than X forwarding: the Pi is driven over
ssh, and forwarding an imshow window adds a synchronous network round trip to
every frame of the control loop. Here the loop hands off the newest frame and
moves on -- if the browser cannot keep up it drops frames instead of stalling
the robot, so watching cannot change how the robot drives.

Open the URL printed at startup. Frames are still written to disk as before, so
a run can be reviewed after the fact too.

Why interrupt rather than os._exit(0): that skipped every finally block, so
main.py's `finally: motors.stop()` never ran and pigpiod held the last PWM duty
forever -- that is how the drivers were found latched at duty 89/191 with both
enables HIGH. KeyboardInterrupt is already handled by main.py, so its normal
cleanup runs. A post-run sweep zeroes the pins even if main.py dies hard.
"""
import os
import socket
import sys
import time
import threading
import _thread
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import cv2

OUT = "/tmp/harness_out"
RUN_SECONDS = float(os.environ.get("HARNESS_SECONDS", "12"))
PORT = int(os.environ.get("HARNESS_PORT", "8088"))
SAVE_FRAMES = int(os.environ.get("HARNESS_SAVE", "40"))

os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT):
    os.remove(os.path.join(OUT, f))

import config
config.SHOW_DEBUG_VIEW = True
config.DEBUG_VIEW_EVERY_N_FRAMES = 1

saved = {}

# Single-slot handoff: only the newest frame matters for a live view, so a slow
# client must never make the control loop wait.
_latest = {"jpeg": None}
_latest_lock = threading.Lock()
_frame_ready = threading.Condition(_latest_lock)
_served = [0]


def mock_imshow(winname, mat):
    key = winname.replace(" ", "_")
    n = saved.get(key, 0)
    if n < SAVE_FRAMES:
        cv2.imwrite(os.path.join(OUT, "%s_%03d.jpg" % (key, n)), mat,
                    [cv2.IMWRITE_JPEG_QUALITY, 70])
    saved[key] = n + 1

    if key == "Camera_+_Decisions":
        ok, buf = cv2.imencode(".jpg", mat, [cv2.IMWRITE_JPEG_QUALITY, 65])
        if ok:
            with _frame_ready:
                _latest["jpeg"] = buf.tobytes()
                _frame_ready.notify_all()


PAGE = b"""<!doctype html><html><head><title>TemuFollower live</title>
<style>body{background:#111;color:#ddd;font-family:monospace;text-align:center;margin:0;padding:12px}
img{max-width:98vw;border:1px solid #444;image-rendering:pixelated}</style></head>
<body><h3>TemuFollower &mdash; live debug view</h3>
<img src="/stream"><p>stream drops frames when the browser lags; the robot never waits</p>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path != "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)
            return

        self.send_response(200)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        _served[0] += 1
        last = None
        try:
            while True:
                with _frame_ready:
                    # Wait for a frame NEWER than the one already sent, so a fast
                    # client does not respin on the same image.
                    if not _frame_ready.wait_for(
                            lambda: _latest["jpeg"] is not None
                            and _latest["jpeg"] is not last, timeout=5.0):
                        if _latest["jpeg"] is None:
                            continue
                    jpeg = _latest["jpeg"]
                last = jpeg
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                 b"Content-Length: " + str(len(jpeg)).encode()
                                 + b"\r\n\r\n" + jpeg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


server = HTTPServer(("0.0.0.0", PORT), Handler)
server.daemon_threads = True
threading.Thread(target=server.serve_forever, daemon=True).start()
print("[harness] LIVE VIEW -> http://%s:%d/" % (lan_ip(), PORT), flush=True)


def stopper():
    time.sleep(RUN_SECONDS)
    print("\n[harness] %.0fs elapsed -> interrupting main thread for clean shutdown"
          % RUN_SECONDS, flush=True)
    _thread.interrupt_main()


if RUN_SECONDS > 0:
    threading.Thread(target=stopper, daemon=True).start()
    print("[harness] running main loop for %.0fs" % RUN_SECONDS, flush=True)
else:
    print("[harness] running main loop until Ctrl-C", flush=True)

rc = 0
try:
    with patch("cv2.imshow", side_effect=mock_imshow), \
         patch("cv2.waitKey", side_effect=lambda delay=0: -1), \
         patch("cv2.destroyAllWindows", side_effect=lambda: None), \
         patch("cv2.namedWindow", side_effect=lambda *a, **k: None), \
         patch("cv2.resizeWindow", side_effect=lambda *a, **k: None):
        import main
        main.main()
except KeyboardInterrupt:
    print("[harness] main loop exited via KeyboardInterrupt (cleanup ran)", flush=True)
except BaseException as e:
    rc = 1
    import traceback
    print("[harness] MAIN CRASHED: %r" % (e,), flush=True)
    traceback.print_exc()

# Independent verification that nothing is still driving. Reads duty cycle, not
# pin level -- level samples a PWM waveform at an instant and is meaningless.
try:
    import pigpio
    from config import (LEFT_LPWM, LEFT_RPWM, RIGHT_LPWM, RIGHT_RPWM,
                        LEFT_EN, RIGHT_EN)
    pi = pigpio.pi()
    if pi.connected:
        pins = [("LEFT_LPWM", LEFT_LPWM), ("LEFT_RPWM", LEFT_RPWM),
                ("RIGHT_LPWM", RIGHT_LPWM), ("RIGHT_RPWM", RIGHT_RPWM)]
        before = [(n, pi.get_PWM_dutycycle(p)) for n, p in pins]
        residual = [nv for nv in before if nv[1]]
        for _, p in pins:
            pi.set_PWM_dutycycle(p, 0)
        pi.write(LEFT_EN, 0)
        pi.write(RIGHT_EN, 0)
        after = [pi.get_PWM_dutycycle(p) for _, p in pins]
        print("\n[harness] duty at exit: %s" % (before,), flush=True)
        if residual:
            print("[harness] WARNING: cleanup left these driving: %s" % (residual,))
        else:
            print("[harness] motors were already idle before sweep (cleanup worked)")
        print("[harness] after sweep: duty=%s EN=%d,%d"
              % (after, pi.read(LEFT_EN), pi.read(RIGHT_EN)), flush=True)
        pi.stop()
except Exception as e:
    print("[harness] could not verify pins: %r" % (e,), flush=True)

print("[harness] frames written: %s" % (saved,), flush=True)
print("[harness] stream clients connected: %d" % _served[0], flush=True)
sys.stdout.flush()
os._exit(rc)
