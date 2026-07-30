"""Push the working tree to the Pi, run the harness there, pull the frames back.

Uses the ssh/scp binaries and the `pi` host alias in ~/.ssh/config rather than
paramiko: paramiko is not installed in this interpreter, and the key auth that
the alias already sets up means there is no password to hand around anyway.

The harness itself streams a live MJPEG view while it runs; its URL is printed
in the output below as soon as the Pi reaches it, so `--seconds 0` plus a
browser is the way to watch the robot drive for as long as you want.
"""
import argparse
import os
import subprocess
import sys

HOST = os.environ.get("PI_HOST", "pi")
# Only used to start pigpiod, which needs root. Override with PI_PASSWORD.
PI_PASSWORD = os.environ.get("PI_PASSWORD", "toor")
REMOTE = "/home/dpsi-lfr/Desktop/Anti-Inertia/TemuFollower"
LOCAL = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(LOCAL, "artifacts", "pi_last")

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", default="12",
                    help="run duration; 0 means run until Ctrl-C")
parser.add_argument("--no-push", action="store_true")
parser.add_argument("--no-pull", action="store_true")
parser.add_argument("--speed-cap", default="1.0",
                    help="scale every forward command; 0.5 = half speed for "
                         "floor testing, 1.0 = normal")
args = parser.parse_args()


def ssh(cmd, **kw):
    return subprocess.run(["ssh", HOST, cmd], **kw)


if not args.no_push:
    print("Pushing .py files to the Pi...", flush=True)
    files = [f for f in os.listdir(LOCAL) if f.endswith(".py")]
    scp = subprocess.run(["scp", "-q"] + [os.path.join(LOCAL, f) for f in files]
                         + ["%s:%s/" % (HOST, REMOTE)])
    if scp.returncode:
        sys.exit("push failed")

# pigpiod owns the PWM peripheral; without it pi_motors cannot connect and
# main.py aborts before the camera ever opens. It is installed under
# /usr/local/bin and started as a bare daemon here, not as a systemd unit.
#
# `sudo pigpiod` needs a tty for its password prompt, so over a non-interactive
# ssh it failed while the old `||` chain still printed success and the run then
# died deep inside PiMotorDriver. Feed the password on stdin and verify after.
print("Ensuring pigpiod is up...", flush=True)
pig = ssh("pgrep -x pigpiod >/dev/null || echo '%s' | sudo -S pigpiod 2>/dev/null; "
          "sleep 1; pgrep -x pigpiod >/dev/null && echo PIGPIOD_OK || echo PIGPIOD_DOWN"
          % PI_PASSWORD, capture_output=True, text=True)
print(pig.stdout.strip(), flush=True)
if "PIGPIOD_OK" not in pig.stdout:
    sys.exit("pigpiod is not running on the Pi -- start it with: sudo pigpiod")

print("Running harness (HARNESS_SECONDS=%s SPEED_CAP=%s)...\n"
      % (args.seconds, args.speed_cap), flush=True)
rc = ssh("cd %s && HARNESS_SECONDS=%s SPEED_CAP=%s realsense-env/bin/python "
         "_scratch_harness.py 2>&1"
         % (REMOTE, args.seconds, args.speed_cap)).returncode

if not args.no_pull:
    print("\nPulling frames...", flush=True)
    os.makedirs(ARTIFACTS, exist_ok=True)
    for f in os.listdir(ARTIFACTS):
        os.remove(os.path.join(ARTIFACTS, f))
    subprocess.run(["scp", "-q", "%s:/tmp/harness_out/*" % HOST, ARTIFACTS])
    print("Frames in %s" % ARTIFACTS)

sys.exit(rc)
