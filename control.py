import time
import math
import random
from collections import deque

class MazeNode:
    """Represents a cell in the maze grid."""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.visited = False
        self.explored_dirs = set()  # dirs we already tried from here: "left", "right", "forward"
        self.parent = None           # for backtracking path

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return isinstance(other, MazeNode) and self.x == other.x and self.y == other.y


class ControlAgent:
    def __init__(self, target_x_center=160, base_speed=1.0, reactive_only=False, steer_invert=True):
        self.target_x = target_x_center
        self.base_speed = base_speed
        self.reactive_only = reactive_only  # If True, only PID follow — no maze memory

        # PID constants
        self.Kp = 0.003
        self.Ki = 0.0001
        self.Kd = 0.002
        self.nl_factor = 1.5

        self.integral = 0
        self.last_error = 0
        self.last_time = time.time()

        # ---- MAZE SOLVING STATE (disabled in reactive_only mode) ----
        self.grid_x = 0
        self.grid_y = 0
        self.heading = 0
        self.nodes = {}
        if not self.reactive_only:
            self._get_or_create_node(0, 0)

        self.explore_stack = []
        self.backtrack_path = deque()
        self.state = "FOLLOWING"
        self.last_known_error = 0
        self.turn_start_time = 0
        self.turn_direction = "forward"
        self.turn_min_duration = 0.0
        self.turn_max_duration = 0.0
        self.turn_left_speed = 0.0
        self.turn_right_speed = 0.0
        self.spin_start_time = 0
        self.backtrack_target = None

        # ---- DASHED LINE HANDLING ----
        self.dashed_countdown = 0

        # ---- MOTOR OUTPUT SANITIZATION ----
        self.MIN_MOTOR_SPEED = 0.15
        self.MAX_MOTOR_SPEED = 1.0

        # ---- RECOVERY (spin search for a lost line) ----
        self.LINE_LOST_STOP_TIMEOUT_S = 4.0
        self.recovery_start_time = None

        # ---- RED LINE STOP ----
        self.RED_STOP_TIMEOUT_S = 2.0
        self._red_stop_until = 0.0

        # ---- APPROACH (wide-ROI reacquire, far/curved line) ----
        self.APPROACH_SPEED = 0.4 * self.base_speed

        # ---- SPEED RAMPING ----
        self.SLEW_RATE_PER_S = 0.3
        self.applied_forward = 0.0
        self._dt = 0.0

        # ---- STEERING: deadzone + blended sharp-turn response ----
        self.CENTER_DEADZONE = 0.12
        self.SHARP_TURN_SPEED = 0.6
        self.TURN_LIMIT = 0.8
        # Second, tighter authority cap applied at the wheel-mixing stage
        # (_apply_steering), separate from TURN_LIMIT above which only
        # bounds the intermediate PID/blend math. Dana/test_PID_camera.py
        # uses this same two-stage design (its own TURN_LIMIT=0.80 +
        # MAX_TURN_SPEED=0.40) specifically because, combined with symmetric
        # reverse-capable steering, letting the full TURN_LIMIT reach the
        # motors turns any moderate line error into a near-full pivot —
        # forward and turn both push each wheel toward its opposite extreme,
        # so the robot whips side to side instead of tracking the line.
        self.MAX_TURN_SPEED = 0.4
        self.MAX_ERROR_PX = float(self.target_x)

        # Dana's hardware testing (Dana/test_PID_camera.py) found the REAL
        # robot's motors turn opposite to the naive "turn>0 -> right slower"
        # model and added a STEER_INVERT flag to compensate. That correction
        # is real-hardware-specific: simulation.py's PyBullet joints have no
        # such quirk (left/right wheel velocities map straight to physical
        # turn direction), so inverting there makes every correction fire
        # backwards — the robot steers away from the line instead of toward
        # it. main.py (real hardware) should keep the default True;
        # simulation.py must pass steer_invert=False.
        self.STEER_INVERT = steer_invert

        # ---- SPECIAL-STATE MEMORY (intersection/dead-end persistence) ----
        # A single frame where vision sees nothing (glare, motion blur, a
        # brief occlusion) shouldn't make the robot act like an intersection
        # or dead-end it just detected never happened. Held briefly so a
        # blank frame doesn't erase it, but cleared as soon as it's acted on.
        self.SPECIAL_STATE_HOLD_S = 0.35
        self._held_special_state = None
        self._held_dead_end = False
        self._held_until = 0.0

        # ---- SPEED PROFILING (fast on straights, slow before turns) ----
        self.CURVE_NORM_MAX = math.radians(60)
        self.CURVE_SLOWDOWN_GAIN = 0.5
        self.ERROR_SLOWDOWN_GAIN = 0.35
        # Extra slowdown tied directly to how hard _pid_follow is currently
        # steering (the same deadzone-"excess" ratio that blends turn toward
        # SHARP_TURN_SPEED) — Dana's hardware testing found forward speed
        # itself needs to collapse toward the sharp-turn response, not just
        # the differential wheel split, or hard turns understeer.
        self.SHARP_TURN_SLOWDOWN_GAIN = 0.6
        self.APPROACH_SLOWDOWN_FACTOR = 0.55
        self.STRAIGHT_BOOST_FACTOR = 1.3
        self.MIN_SPEED_SCALE = 0.15
        self.MAX_SPEED_SCALE = 1.3

        # ---- TRUE PIVOT TURNING (reverse-capable) ----
        # Fractions of base_speed used for in-place pivots (one wheel
        # forward, one wheel reverse) instead of a forward-only differential
        # — merged from Dana/test_PID_camera.py, which relies on real
        # reverse output for its SPIN_SEARCH state and (via its symmetric
        # forward+/-turn mix) for any sharp turn where turn exceeds forward.
        self.TURN_PIVOT_SPEED = 0.6
        self.SPIN_PIVOT_SPEED = 0.5
        self.RECOVERY_PIVOT_SPEED = 0.45

        # ---- SELF-TUNING PID ----
        self.RELAY_AMPLITUDE = 0.35
        self.AUTOTUNE_BASE_SPEED = 0.5 * self.base_speed
        self.AUTOTUNE_CYCLES = 4
        self.AUTOTUNE_TIMEOUT_S = 20.0
        self.autotune_start_time = 0.0
        self.autotune_last_sign = 0
        self.autotune_half_cycle_peak = 0.0
        self.autotune_zero_crossings = []
        self.autotune_peaks = []

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _get_or_create_node(self, gx, gy):
        key = (gx, gy)
        if key not in self.nodes:
            self.nodes[key] = MazeNode(gx, gy)
        return self.nodes[key]

    def _current_node(self):
        return self._get_or_create_node(self.grid_x, self.grid_y)

    def _move_grid(self, direction):
        """Advance grid coords one step in the given direction (relative to current heading)."""
        actual = (self.heading + {"forward": 0, "right": 1, "backward": 2, "left": 3}[direction]) % 4
        dx = [0, 1, 0, -1][actual]
        dy = [1, 0, -1, 0][actual]
        self.grid_x += dx
        self.grid_y += dy

    def _dir_from_vision_error(self, error):
        """Map PID error to a direction name."""
        if error < -30:
            return "left"
        elif error > 30:
            return "right"
        else:
            return "forward"

    def _direction_leads_to_visited(self, node, direction):
        """Whether taking `direction` from `node` (given current heading) walks
        straight back onto a node we've already visited."""
        actual = (self.heading + {"forward": 0, "right": 1, "backward": 2, "left": 3}[direction]) % 4
        dx = [0, 1, 0, -1][actual]
        dy = [1, 0, -1, 0][actual]
        neighbor = self.nodes.get((node.x + dx, node.y + dy))
        return neighbor is not None and neighbor.visited

    def _pick_unexplored_direction(self, node):
        """Return a direction string not yet explored from this node, or None.

        Prefers a direction that leads into unvisited territory — the robot
        shouldn't want to re-tread ground it's already covered. Only falls
        back to a direction that leads to a visited node if every unexplored
        option does (e.g. a loop back to an earlier intersection).
        """
        candidates = ["left", "right", "forward"]
        random.shuffle(candidates)
        unexplored = [d for d in candidates if d not in node.explored_dirs]
        if not unexplored:
            return None

        fresh = [d for d in unexplored if not self._direction_leads_to_visited(node, d)]
        return fresh[0] if fresh else unexplored[0]

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------
    def calculate_speeds(self, vision_data):
        left, right = self._calculate_speeds_raw(vision_data)
        return self._sanitize_speeds(left, right)

    def _sanitize_speeds(self, left, right):
        """Clamp motor outputs to a valid, physically meaningful range.

        Values are clamped to [-MAX_MOTOR_SPEED, MAX_MOTOR_SPEED] — negative
        values are real reverse output (hardware.py's _set_motor already
        supports it), needed for true in-place pivots on sharp turns and
        spins, matching Dana/test_PID_camera.py's hardware-validated
        reverse-capable steering. Any nonzero magnitude below MIN_MOTOR_SPEED
        is floored to it (sign preserved) so the motors don't stall below
        their deadband.
        """
        def clean(speed):
            speed = max(-self.MAX_MOTOR_SPEED, min(self.MAX_MOTOR_SPEED, speed))
            if 0.0 < abs(speed) < self.MIN_MOTOR_SPEED:
                speed = self.MIN_MOTOR_SPEED if speed > 0 else -self.MIN_MOTOR_SPEED
            return speed

        return clean(left), clean(right)

    def _calculate_speeds_raw(self, vision_data):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        if dt <= 0:
            dt = 0.001
        self._dt = dt

        # ---- state dispatch ----
        if self.state == "STOPPED":
            return 0.0, 0.0

        if self.state == "AUTOTUNE":
            return self._handle_autotune(vision_data, current_time)

        if self.state == "TURNING":
            return self._handle_turn(vision_data, current_time)

        if self.state == "SPINNING":
            return self._handle_spin(current_time)

        if self.state == "BACKTRACKING":
            return self._handle_backtrack(vision_data, current_time)

        if self.state == "RECOVERING":
            return self._handle_recovery(vision_data, current_time)

        if self.state == "RED_STOP":
            if current_time < self._red_stop_until:
                return 0.0, 0.0
            self.state = "FOLLOWING"
            self.applied_forward = 0.0

        # ---- FOLLOWING ----
        cx = vision_data.get("line_center_x")
        cy = vision_data.get("line_center_y")
        line_ended = vision_data.get("line_ended", False)
        is_dashed = vision_data.get("is_dashed", False)
        special = vision_data.get("special_state")
        curvature = vision_data.get("line_curvature", 0.0)

        # --- Special-state memory: a single blank frame (glare, motion blur,
        # brief occlusion) right after seeing an intersection/dead-end
        # shouldn't make the robot act like it never happened. Only kicks in
        # when vision is truly blank this frame — a fresh, resolved line
        # detection always wins over a stale hold. ---
        if special in ("intersection", "dead_end"):
            self._held_special_state = special
            self._held_dead_end = line_ended
            self._held_until = current_time + self.SPECIAL_STATE_HOLD_S
        elif cx is None and special is None and current_time < self._held_until:
            special = self._held_special_state
            line_ended = self._held_dead_end

        # --- Dashed line: keep driving straight for a few frames ---
        if is_dashed and self.dashed_countdown <= 0:
            self.dashed_countdown = 8  # ~8 frames of straight driving through gap

        if self.dashed_countdown > 0:
            self.dashed_countdown -= 1
            # Continue straight regardless of line data
            if cx is not None:
                return self._pid_follow(cx, curvature, special)
            forward = self._slew_forward(self.base_speed)
            return forward, forward  # straight

        # --- Red line: stop motors for RED_STOP_TIMEOUT_S ---
        if special == "red_line" or vision_data.get("red_line_detected"):
            self.state = "RED_STOP"
            self._red_stop_until = current_time + self.RED_STOP_TIMEOUT_S
            self._held_until = 0.0
            print(f"[Control] Red line stop — {self.RED_STOP_TIMEOUT_S}s hold")
            return 0.0, 0.0

        # --- Wide-ROI reacquire: line is far/curved but still visible ---
        if special == "approach" and cx is not None:
            self.state = "APPROACH"
            return self._approach_speeds(cx)

        # --- Line ended (dead end) ---
        if line_ended and special == "dead_end":
            self._held_until = 0.0
            node = self._current_node()
            node.visited = True
            # Mark forward as explored (it's a dead end)
            node.explored_dirs.add("forward")
            print(f"[Maze] Dead end at ({self.grid_x},{self.grid_y}). Initiating 180 + backtrack.")
            self._start_spin(current_time)
            return 0.0, 0.0  # will be overridden next frame

        # --- Intersection detected ---
        if special == "intersection":
            self._held_until = 0.0
            node = self._current_node()
            if not node.visited:
                node.visited = True
                print(f"[Maze] Intersection at ({self.grid_x},{self.grid_y}). "
                      f"Explored so far: {node.explored_dirs}")
            # Try unexplored direction
            unexplored = self._pick_unexplored_direction(node)
            if unexplored is not None:
                node.explored_dirs.add(unexplored)
                print(f"[Maze] Turning {unexplored.upper()} at ({self.grid_x},{self.grid_y})")
                self._start_turn(unexplored, current_time)
                return 0.0, 0.0
            else:
                # All directions explored — backtrack
                print(f"[Maze] All explored at ({self.grid_x},{self.grid_y}). Backtracking.")
                self.state = "BACKTRACKING"
                self._setup_backtrack()
                return 0.0, 0.0

        # --- Normal PID line following ---
        if cx is not None:
            self.state = "FOLLOWING"
            return self._pid_follow(cx, curvature, special)

        # No line and not a dead end — recovery spin
        self._enter_recovery(current_time)
        return self._handle_recovery(vision_data, current_time)

    # ------------------------------------------------------------------
    # Steering output (shared by all states so STEER_INVERT only lives here)
    # ------------------------------------------------------------------
    def _apply_steering(self, forward, turn):
        """Convert a forward speed + signed turn command into left/right
        wheel speeds using symmetric tank-steering (left=forward+turn,
        right=forward-turn) — both wheels move, roughly double the turning
        authority of a forward-only differential, and the inner wheel goes
        negative (true reverse pivot) once turn exceeds forward on a sharp
        turn. Matches Dana/test_PID_camera.py's hardware-validated mix.
        turn>0 means "steer right" in the un-inverted convention (right
        wheel slows/reverses). STEER_INVERT flips that mapping to match what
        Dana's hardware testing found the real motors do.

        `turn` is re-clamped to MAX_TURN_SPEED here — tighter than the
        TURN_LIMIT already applied upstream — so this single mixing point is
        the actual final authority cap for every caller, not just PID
        follow's blend logic."""
        if self.STEER_INVERT:
            turn = -turn
        turn = max(-self.MAX_TURN_SPEED, min(self.MAX_TURN_SPEED, turn))
        left = forward + turn
        right = forward - turn
        return left, right

    def _compute_speed_scale(self, curvature, norm_error, special_state):
        """Scale base_speed up on flat, straight sections and down before a
        curve or intersection, using the snake tracker's lookahead curvature
        (vision.py) and its early "approaching_intersection" warning — so
        the robot starts slowing before it reaches the turn, not once it's
        already on top of it.

        Also discounts by the same deadzone-"excess" ratio _pid_follow uses
        to blend turn toward SHARP_TURN_SPEED, so the forward speed itself
        collapses on a genuinely sharp turn rather than only the raw
        error-magnitude term above — Dana's hardware-validated design ties
        these together (its target_forward shrinks with the same blend that
        drives target_turn toward its sharp-turn value)."""
        curve_norm = max(0.0, min(1.0, curvature / self.CURVE_NORM_MAX)) if self.CURVE_NORM_MAX else 0.0
        error_norm = min(1.0, abs(norm_error))
        excess = (error_norm - self.CENTER_DEADZONE) / (1.0 - self.CENTER_DEADZONE) if self.CENTER_DEADZONE < 1.0 else 0.0
        excess = max(0.0, min(1.0, excess))

        scale = (1.0 - self.CURVE_SLOWDOWN_GAIN * curve_norm
                 - self.ERROR_SLOWDOWN_GAIN * error_norm
                 - self.SHARP_TURN_SLOWDOWN_GAIN * excess)

        if special_state == "approaching_intersection":
            scale *= self.APPROACH_SLOWDOWN_FACTOR

        if special_state is None and curve_norm < 0.05 and error_norm < 0.05:
            scale = max(scale, self.STRAIGHT_BOOST_FACTOR)

        return max(self.MIN_SPEED_SCALE, min(self.MAX_SPEED_SCALE, scale))

    # ------------------------------------------------------------------
    # PID line following
    # ------------------------------------------------------------------
    def _pid_follow(self, cx, curvature=0.0, special_state=None):
        error = cx - self.target_x
        self.last_known_error = error
        norm_error = error / self.MAX_ERROR_PX if self.MAX_ERROR_PX else 0.0

        sign = 1 if error > 0 else -1
        p_error = sign * (abs(error) ** self.nl_factor)

        dt = self._dt
        self.integral += error * dt
        self.integral = max(-1000, min(1000, self.integral))
        derivative = (error - self.last_error) / dt
        self.last_error = error

        pid_turn = (self.Kp * p_error) + (self.Ki * self.integral) + (self.Kd * derivative)

        if abs(norm_error) <= self.CENTER_DEADZONE:
            turn = pid_turn
        else:
            # Beyond the deadzone, blend the PID turn toward a fixed sharp-turn
            # response so large errors react decisively instead of waiting on
            # the (comparatively slow) integral/derivative terms to catch up.
            excess = (abs(norm_error) - self.CENTER_DEADZONE) / (1.0 - self.CENTER_DEADZONE)
            excess = max(0.0, min(1.0, excess))
            sharp_turn = sign * self.SHARP_TURN_SPEED
            turn = (1.0 - excess) * pid_turn + excess * sharp_turn

        turn = max(-self.TURN_LIMIT, min(self.TURN_LIMIT, turn))

        speed_scale = self._compute_speed_scale(curvature, norm_error, special_state)
        forward = self._slew_forward(self.base_speed * speed_scale)

        return self._apply_steering(forward, turn)

    # ------------------------------------------------------------------
    # Approach (wide-ROI reacquire — line far away or around a curve)
    # ------------------------------------------------------------------
    def _approach_speeds(self, cx):
        """Gentle proportional steering at reduced speed while closing back
        in on a line that was only found via the wide/far search ROI."""
        error = cx - self.target_x
        self.last_known_error = error

        Kp_approach = 0.006
        turn = Kp_approach * error
        turn = max(-self.TURN_LIMIT, min(self.TURN_LIMIT, turn))

        forward = self._slew_forward(self.APPROACH_SPEED)
        return self._apply_steering(forward, turn)

    # ------------------------------------------------------------------
    # Speed ramping
    # ------------------------------------------------------------------
    def _slew_forward(self, target_forward):
        """Limit how fast the forward speed component can change per second,
        so the robot ramps smoothly instead of snapping to full speed."""
        max_delta = self.SLEW_RATE_PER_S * self._dt
        if target_forward > self.applied_forward:
            self.applied_forward = min(target_forward, self.applied_forward + max_delta)
        else:
            self.applied_forward = max(target_forward, self.applied_forward - max_delta)
        return self.applied_forward

    # ------------------------------------------------------------------
    # Turn handling (intersection turns)
    # ------------------------------------------------------------------
    def _turn_speeds_for_direction(self, direction):
        """Un-inverted convention: turning left means the left wheel reverses
        and the right wheel drives forward — a true in-place pivot instead of
        a forward-only differential, matching Dana's reverse-capable
        hardware testing. STEER_INVERT swaps which physical side that maps
        to, same as it does for PID steering — kept as one flag so turn
        direction and line-following steering can never disagree."""
        if direction == "forward":
            return self.base_speed, self.base_speed

        pivot = self.TURN_PIVOT_SPEED * self.base_speed
        if direction == "left":
            left_speed, right_speed = -pivot, pivot
        else:  # right
            left_speed, right_speed = pivot, -pivot

        if self.STEER_INVERT:
            left_speed, right_speed = right_speed, left_speed
        return left_speed, right_speed

    def _start_turn(self, direction, current_time):
        self.state = "TURNING"
        self.turn_direction = direction
        self.turn_start_time = current_time
        # A brief blind minimum so the robot actually clears the wide
        # intersection line before checking vision again, then a generous
        # cap as a fallback if the line is never reconfirmed.
        self.turn_min_duration = 0.3 if direction == "forward" else 0.35
        self.turn_max_duration = 0.3 if direction == "forward" else 1.4
        self.turn_left_speed, self.turn_right_speed = self._turn_speeds_for_direction(direction)

    def _handle_turn(self, vision_data, current_time):
        elapsed = current_time - self.turn_start_time
        if elapsed >= self.turn_min_duration:
            cx = vision_data.get("line_center_x")
            special = vision_data.get("special_state")
            # A line that's reappeared, roughly centered, and not still
            # reading as an intersection means the turn has actually
            # completed — don't wait out the full timer if it's not needed.
            line_reacquired = (
                cx is not None
                and special in (None, "approach")
                and abs(cx - self.target_x) < self.MAX_ERROR_PX * 0.35
            )
            if line_reacquired or elapsed >= self.turn_max_duration:
                self._move_grid(self.turn_direction)
                self.state = "FOLLOWING"
                return self.base_speed, self.base_speed
        return self.turn_left_speed, self.turn_right_speed

    # ------------------------------------------------------------------
    # Spin handling (dead end 180)
    # ------------------------------------------------------------------
    def _start_spin(self, current_time):
        self.state = "SPINNING"
        self.spin_start_time = current_time

    def _handle_spin(self, current_time):
        elapsed = current_time - self.spin_start_time
        spin_duration = 1.2  # time for 180-degree spin
        if elapsed < spin_duration:
            # True in-place pivot (left reverse, right forward), scaled by
            # base_speed — direction is arbitrary for a 180 (heading update
            # below is an unconditional +2 either way), so no STEER_INVERT
            # swap is needed here, unlike the direction-sensitive turns.
            pivot = self.SPIN_PIVOT_SPEED * self.base_speed
            return -pivot, pivot
        else:
            # Spin complete — update heading and start backtracking
            self.heading = (self.heading + 2) % 4  # 180 degrees
            self.state = "BACKTRACKING"
            self._setup_backtrack()
            return self.base_speed, self.base_speed

    # ------------------------------------------------------------------
    # Backtracking (return to last intersection, try other route)
    # ------------------------------------------------------------------
    def _setup_backtrack(self):
        """
        Build a path from current node back to the nearest unvisited intersection
        using parent pointers (BFS from current node, or walk back through stack).
        """
        start = self._current_node()

        # BFS to find nearest node that still has unexplored directions
        visited_bfs = set()
        queue = deque()
        queue.append((start, []))
        visited_bfs.add((start.x, start.y))

        while queue:
            node, path = queue.popleft()
            unexplored = self._pick_unexplored_direction(node)
            if unexplored is not None and len(path) > 0:
                # Found a node with unexplored directions — backtrack to it
                self.backtrack_path = deque(path)
                self.backtrack_target = node
                print(f"[Maze] Backtracking to ({node.x},{node.y}) — {len(path)} steps away")
                return

            # Explore neighbors (forward, left, right relative to the grid)
            for dx, dy, dname in [(0, 1, "forward"), (1, 0, "right"), (0, -1, "backward"), (-1, 0, "left")]:
                nx, ny = node.x + dx, node.y + dy
                if (nx, ny) not in visited_bfs and (nx, ny) in self.nodes:
                    visited_bfs.add((nx, ny))
                    neighbor = self.nodes[(nx, ny)]
                    new_path = path + [(neighbor, dname)]
                    queue.append((neighbor, new_path))

        # No unexplored intersections found — maze fully explored, stop
        print("[Maze] All intersections fully explored! Maze solved. Stopping.")
        self.state = "STOPPED"

    def _handle_backtrack(self, vision_data, current_time):
        """Drive straight / PID follow while navigating back along the backtrack path."""
        if not self.backtrack_path:
            # Reached the target intersection — now pick an unexplored direction
            if self.backtrack_target is not None:
                node = self.backtrack_target
                unexplored = self._pick_unexplored_direction(node)
                if unexplored is not None:
                    node.explored_dirs.add(unexplored)
                    print(f"[Maze] Arrived at ({node.x},{node.y}). Turning {unexplored.upper()}")
                    self._start_turn(unexplored, current_time)
                    return 0.0, 0.0
            self.state = "FOLLOWING"
            return self.base_speed, self.base_speed

        # Follow the line toward the next backtrack node
        cx = vision_data.get("line_center_x")
        if cx is not None:
            return self._pid_follow(cx, vision_data.get("line_curvature", 0.0), vision_data.get("special_state"))
        return self.base_speed, self.base_speed

    # ------------------------------------------------------------------
    # Recovery (lost line, spinning to find it)
    # ------------------------------------------------------------------
    def _enter_recovery(self, current_time):
        if self.state != "RECOVERING":
            self.recovery_start_time = current_time
        self.state = "RECOVERING"

    def _recovered_to_following(self):
        """Line reacquired after being genuinely lost. Rather than blindly
        continuing wherever the spin happened to point, head back to the
        nearest intersection with unexplored options — the robot shouldn't
        assume whatever's ahead is new ground just because it found a line."""
        print("[Maze] Line reacquired after recovery — returning to nearest open intersection.")
        self.applied_forward = 0.0
        self.state = "BACKTRACKING"
        self._setup_backtrack()

    def _handle_recovery(self, vision_data, current_time):
        cx = vision_data.get("line_center_x")
        if cx is not None:
            self._recovered_to_following()
            return self.base_speed, self.base_speed

        if (self.recovery_start_time is not None
                and (current_time - self.recovery_start_time) > self.LINE_LOST_STOP_TIMEOUT_S):
            # Give up spinning in place, but keep watching for the line —
            # _calculate_speeds_raw re-enters this every frame regardless.
            return 0.0, 0.0

        # Spin toward whichever side the line was last seen on. In the
        # un-inverted convention a positive error (line to the right) means
        # spin right (right wheel reverses, left wheel forward) — STEER_INVERT
        # flips which physical side that is, same as everywhere else. True
        # in-place pivot (both wheels driven, opposite signs) instead of one
        # wheel idle, matching Dana's SPIN_SEARCH design.
        spin_toward_positive_error = self.last_known_error > 0
        if self.STEER_INVERT:
            spin_toward_positive_error = not spin_toward_positive_error

        pivot = self.RECOVERY_PIVOT_SPEED * self.base_speed
        if spin_toward_positive_error:
            return pivot, -pivot
        else:
            return -pivot, pivot

    # ------------------------------------------------------------------
    # Self-tuning PID (relay-based / Ziegler-Nichols auto-tune)
    # ------------------------------------------------------------------
    def start_autotune(self):
        """Kick off a relay auto-tune run. Drives a bang-bang steering signal
        to force the line-error into a sustained oscillation, measures its
        period and amplitude, and derives new Kp/Ki/Kd from them."""
        print("[Autotune] Starting relay-based PID auto-tune...")
        self.state = "AUTOTUNE"
        self.autotune_start_time = time.time()
        self.autotune_last_sign = 0
        self.autotune_half_cycle_peak = 0.0
        self.autotune_zero_crossings = []
        self.autotune_peaks = []

    def _handle_autotune(self, vision_data, current_time):
        cx = vision_data.get("line_center_x")
        if cx is None:
            print("[Autotune] Lost the line during auto-tune, aborting. Keeping previous gains.")
            self._enter_recovery(current_time)
            return self._handle_recovery(vision_data, current_time)

        error = cx - self.target_x
        # Same nonlinear transform _pid_follow applies before multiplying by
        # Kp, so the identified gain lines up with how it will actually be used.
        sign_now = 1 if error > 0 else (-1 if error < 0 else 0)
        p_error = sign_now * (abs(error) ** self.nl_factor)

        self.autotune_half_cycle_peak = max(self.autotune_half_cycle_peak, abs(p_error))

        sign = sign_now if sign_now != 0 else self.autotune_last_sign
        if self.autotune_last_sign != 0 and sign != 0 and sign != self.autotune_last_sign:
            # Error crossed zero — a half-cycle of the forced oscillation just ended.
            self.autotune_zero_crossings.append(current_time)
            self.autotune_peaks.append(self.autotune_half_cycle_peak)
            self.autotune_half_cycle_peak = 0.0
        if sign != 0:
            self.autotune_last_sign = sign

        half_cycles_needed = self.AUTOTUNE_CYCLES * 2
        timed_out = (current_time - self.autotune_start_time) > self.AUTOTUNE_TIMEOUT_S

        if len(self.autotune_zero_crossings) >= half_cycles_needed + 1 or timed_out:
            self._finish_autotune(timed_out)
            return self.base_speed, self.base_speed

        # Relay/bang-bang controller: full deflection based only on error sign,
        # which is what forces the sustained oscillation the tuner measures.
        turn = self.RELAY_AMPLITUDE if error >= 0 else -self.RELAY_AMPLITUDE
        left = self.AUTOTUNE_BASE_SPEED + turn
        right = self.AUTOTUNE_BASE_SPEED - turn
        return left, right

    def _finish_autotune(self, timed_out):
        min_needed = self.AUTOTUNE_CYCLES * 2
        if timed_out or len(self.autotune_zero_crossings) < min_needed:
            print("[Autotune] Not enough oscillation detected, keeping existing PID gains.")
            self.state = "FOLLOWING"
            return

        crossings = self.autotune_zero_crossings[-(min_needed):]
        half_periods = [t2 - t1 for t1, t2 in zip(crossings[:-1], crossings[1:])]
        peaks = self.autotune_peaks[-(min_needed):]

        if not half_periods or not peaks:
            print("[Autotune] Invalid oscillation data, keeping existing PID gains.")
            self.state = "FOLLOWING"
            return

        Tu = 2.0 * (sum(half_periods) / len(half_periods))  # full oscillation period
        a = sum(peaks) / len(peaks)                          # oscillation amplitude

        if a <= 0 or Tu <= 0:
            print("[Autotune] Invalid oscillation data, keeping existing PID gains.")
            self.state = "FOLLOWING"
            return

        Ku = (4.0 * self.RELAY_AMPLITUDE) / (math.pi * a)  # ultimate gain

        # Classic Ziegler-Nichols PID tuning rules from Ku/Tu.
        new_kp = 0.6 * Ku
        new_ki = 2.0 * new_kp / Tu
        new_kd = new_kp * Tu / 8.0

        # Guard against a noisy identification producing runaway gains.
        new_kp = max(0.0001, min(0.05, new_kp))
        new_ki = max(0.0, min(0.005, new_ki))
        new_kd = max(0.0, min(0.05, new_kd))

        print(f"[Autotune] Ku={Ku:.6f} Tu={Tu:.3f}s amplitude={a:.2f} -> "
              f"Kp={new_kp:.6f} Ki={new_ki:.6f} Kd={new_kd:.6f}")

        self.Kp, self.Ki, self.Kd = new_kp, new_ki, new_kd
        self.integral = 0.0
        self.last_error = 0.0
        self.state = "FOLLOWING"
