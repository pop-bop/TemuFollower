#!/usr/bin/env bash
# One-time setup so the launcher works from a double-click.
#
# Two things need doing that run.sh cannot do for itself:
#   * a passwordless sudoers rule for pigpiod -- a double-clicked launcher has
#     no tty, so a sudo password prompt has nowhere to appear and the run dies
#     before the camera opens. The rule is scoped to the single pigpiod binary,
#     not blanket NOPASSWD.
#   * the .desktop file marked executable and trusted, or the file manager
#     shows it as a text file instead of running it.
#
# Run once, on the Pi:  ./install_launcher.sh
set -eu

cd "$(dirname "$(readlink -f "$0")")"

PIGPIOD="$(command -v pigpiod || echo /usr/local/bin/pigpiod)"
USER_NAME="$(id -un)"
RULE="$USER_NAME ALL=(root) NOPASSWD: $PIGPIOD"
RULE_FILE=/etc/sudoers.d/pigpiod-nopasswd

echo "Granting passwordless sudo for: $PIGPIOD"
# visudo -cf validates before install: a malformed sudoers file locks the user
# out of sudo entirely, so it is never written directly.
echo "$RULE" | sudo tee "$RULE_FILE" >/dev/null
sudo chmod 0440 "$RULE_FILE"
if sudo visudo -cf "$RULE_FILE" >/dev/null; then
    echo "  sudoers rule installed and validated"
else
    sudo rm -f "$RULE_FILE"
    echo "  ERROR: rule failed validation, removed" >&2
    exit 1
fi

chmod +x run.sh
echo "run.sh marked executable"

DESKTOP="$HOME/Desktop/LineFollower.desktop"
sed "s#^Exec=.*#Exec=$PWD/run.sh#; s#^Path=.*#Path=$PWD#" \
    LineFollower.desktop > "$DESKTOP"
chmod +x "$DESKTOP"
gio set "$DESKTOP" metadata::trusted true 2>/dev/null || true
echo "Desktop icon installed at $DESKTOP"

echo
echo "Done. Double-click 'Line Follower' on the desktop to run."
