#!/usr/bin/env bash
# One-click launcher for the line follower.
#
# Handles the three things that have to be true before main.py can run, in
# order, and refuses to continue if any of them is not:
#   1. pigpiod is up      -- it owns the PWM peripheral; without it pi_motors
#                            cannot connect and main.py dies after the camera
#                            is already open.
#   2. the venv exists    -- pyrealsense2 lives there, not in system python.
#   3. the motors are idle on exit -- main.py has its own cleanup, but if it
#                            dies hard the PWM stays latched at the last duty
#                            and the robot drives off on its own.
#
# Run from a terminal or by double-clicking TemuFollower.desktop.
set -u

cd "$(dirname "$(readlink -f "$0")")" || exit 1

VENV_PY="realsense-env/bin/python"
RED=$'\e[31m'; GREEN=$'\e[32m'; YELLOW=$'\e[33m'; RESET=$'\e[0m'

fail() { printf '%s\n' "${RED}ERROR: $*${RESET}" >&2; }

# --- 1. pigpiod -----------------------------------------------------------
# Installed under /usr/local/bin and started as a bare daemon, not a systemd
# unit. `sudo pigpiod` wants a tty for its prompt, so when this is launched by
# double-click there is nowhere to type: prefer a passwordless sudoers rule
# (see install_launcher.sh), fall back to prompting in the terminal.
if pgrep -x pigpiod >/dev/null; then
    printf '%s\n' "${GREEN}pigpiod already running${RESET}"
else
    printf '%s\n' "starting pigpiod..."
    if sudo -n pigpiod 2>/dev/null; then
        :
    else
        printf '%s\n' "${YELLOW}sudo needs a password (run install_launcher.sh once to avoid this)${RESET}"
        sudo pigpiod
    fi
    sleep 1
    if pgrep -x pigpiod >/dev/null; then
        printf '%s\n' "${GREEN}pigpiod started${RESET}"
    else
        fail "pigpiod failed to start. Try manually: sudo pigpiod"
        read -rp "Press Enter to close..."
        exit 1
    fi
fi

# --- 2. venv --------------------------------------------------------------
if [ ! -x "$VENV_PY" ]; then
    fail "venv python not found at $PWD/$VENV_PY"
    read -rp "Press Enter to close..."
    exit 1
fi

# --- 3. run, then guarantee the motors are off ----------------------------
# The sweep runs on every exit path -- clean quit, crash, or Ctrl-C -- because
# a latched PWM duty is the difference between a stopped robot and one that
# drives into a wall while nobody is looking.
sweep_motors() {
    "$VENV_PY" - <<'PY' 2>/dev/null
try:
    import pigpio
    from config import LEFT_LPWM, LEFT_RPWM, RIGHT_LPWM, RIGHT_RPWM, LEFT_EN, RIGHT_EN
    pi = pigpio.pi()
    if pi.connected:
        for p in (LEFT_LPWM, LEFT_RPWM, RIGHT_LPWM, RIGHT_RPWM):
            pi.set_PWM_dutycycle(p, 0)
        pi.write(LEFT_EN, 0)
        pi.write(RIGHT_EN, 0)
        pi.stop()
        print("motors swept to idle")
except Exception as e:
    print("could not sweep motors: %r" % (e,))
PY
}
trap sweep_motors EXIT INT TERM

printf '%s\n' "${GREEN}starting line follower  (Ctrl-C to stop)${RESET}"
printf '%s\n' "speed cap: ${SPEED_CAP:-1.0}   (set SPEED_CAP=0.5 for slow floor tests)"
echo
"$VENV_PY" main.py
rc=$?

echo
if [ $rc -ne 0 ]; then
    fail "exited with code $rc"
else
    printf '%s\n' "${GREEN}exited cleanly${RESET}"
fi

# Double-clicked runs get their terminal closed the instant the process ends,
# taking the traceback with it. Hold it open so the error is readable.
if [ -t 0 ]; then
    read -rp "Press Enter to close..."
fi
exit $rc
