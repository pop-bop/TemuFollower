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
    def __init__(self, target_x_center=160, base_speed=1.0):
        self.target_x = target_x_center
        self.base_speed = base_speed

        # PID constants
        self.Kp = 0.003
        self.Ki = 0.0001
        self.Kd = 0.002
        self.nl_factor = 1.5

        self.integral = 0
        self.last_error = 0
        self.last_time = time.time()

        # ---- MAZE SOLVING STATE ----
        # Current position in grid coordinates (start at 0,0)
        self.grid_x = 0
        self.grid_y = 0
        # Heading: 0=forward(+Y), 1=right(+X), 2=backward(-Y), 3=left(-X)
        self.heading = 0

        # Node map: (grid_x, grid_y) -> MazeNode
        self.nodes = {}
        self._get_or_create_node(0, 0)

        # Stack for DFS exploration (micromouse flood-fill style)
        self.explore_stack = []  # list of (node, direction_we_came_from)
        self.backtrack_path = deque()  # path to follow when backtracking

        # State machine
        self.state = "FOLLOWING"
        # FOLLOWING | RED_TRACKING | TURNING | SPINNING | BACKTRACKING | RECOVERING | STOPPED

        self.last_known_error = 0
        self.turn_start_time = 0
        self.turn_duration = 0.0
        self.turn_left_speed = 0.0
        self.turn_right_speed = 0.0

        self.spin_start_time = 0

        self.backtrack_target = None  # next node to move toward while backtracking

        # ---- RED LINE TRACKING ----
        self.saw_red = False

        # ---- DASHED LINE HANDLING ----
        self.dashed_countdown = 0  # frames to keep driving straight during a dashed gap

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

    def _pick_unexplored_direction(self, node):
        """Return a direction string not yet explored from this node, or None."""
        candidates = ["left", "right", "forward"]
        random.shuffle(candidates)
        for d in candidates:
            if d not in node.explored_dirs:
                return d
        return None

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------
    def calculate_speeds(self, vision_data):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        if dt <= 0:
            dt = 0.001

        # ---- state dispatch ----
        if self.state == "STOPPED":
            return 0.0, 0.0

        if self.state == "TURNING":
            return self._handle_turn(current_time)

        if self.state == "SPINNING":
            return self._handle_spin(current_time)

        if self.state == "BACKTRACKING":
            return self._handle_backtrack(vision_data, current_time)

        if self.state == "RECOVERING":
            return self._handle_recovery()

        # ---- FOLLOWING / RED_TRACKING ----
        cx = vision_data.get("line_center_x")
        cy = vision_data.get("line_center_y")
        red_detected = vision_data.get("red_line_detected", False)
        red_cx = vision_data.get("red_line_center_x")
        line_ended = vision_data.get("line_ended", False)
        is_dashed = vision_data.get("is_dashed", False)
        special = vision_data.get("special_state")

        # --- Dashed line: keep driving straight for a few frames ---
        if is_dashed and self.dashed_countdown <= 0:
            self.dashed_countdown = 8  # ~8 frames of straight driving through gap

        if self.dashed_countdown > 0:
            self.dashed_countdown -= 1
            # Continue straight regardless of line data
            if red_detected and red_cx is not None:
                self.state = "RED_TRACKING"
                return self._red_track_speeds(vision_data)
            if cx is not None:
                return self._pid_follow(cx)
            return self.base_speed, self.base_speed  # straight

        # --- Red line tracking ---
        if red_detected and red_cx is not None:
            self.state = "RED_TRACKING"
            return self._red_track_speeds(vision_data)

        # --- Line ended (dead end) ---
        if line_ended and special == "dead_end":
            node = self._current_node()
            node.visited = True
            # Mark forward as explored (it's a dead end)
            node.explored_dirs.add("forward")
            print(f"[Maze] Dead end at ({self.grid_x},{self.grid_y}). Initiating 180 + backtrack.")
            self._start_spin(current_time)
            return 0.0, 0.0  # will be overridden next frame

        # --- Intersection detected ---
        if special == "intersection":
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
            return self._pid_follow(cx)

        # No line and not a dead end — recovery spin
        self.state = "RECOVERING"
        return self._handle_recovery()

    # ------------------------------------------------------------------
    # PID line following
    # ------------------------------------------------------------------
    def _pid_follow(self, cx):
        error = cx - self.target_x
        self.last_known_error = error

        sign = 1 if error > 0 else -1
        p_error = sign * (abs(error) ** self.nl_factor)

        dt = max(time.time() - self.last_time, 0.001)
        self.integral += error * dt
        self.integral = max(-1000, min(1000, self.integral))
        derivative = (error - self.last_error) / dt
        self.last_error = error

        turn = (self.Kp * p_error) + (self.Ki * self.integral) + (self.Kd * derivative)

        left = self.base_speed
        right = self.base_speed
        if turn > 0:
            right -= turn
        else:
            left += turn

        if abs(error) > 50:
            left *= 0.8
            right *= 0.8

        return left, right

    # ------------------------------------------------------------------
    # Red line tracking
    # ------------------------------------------------------------------
    def _red_track_speeds(self, vision_data):
        """Steer toward the red line using proportional control."""
        red_cx = vision_data.get("red_line_center_x")
        if red_cx is None:
            self.state = "FOLLOWING"
            return self.base_speed, self.base_speed

        error = red_cx - self.target_x
        Kp_red = 0.005
        turn = Kp_red * error

        left = self.base_speed * 0.9
        right = self.base_speed * 0.9
        if turn > 0:
            right -= turn
        else:
            left += turn

        left = max(0.0, min(1.0, left))
        right = max(0.0, min(1.0, right))
        return left, right

    # ------------------------------------------------------------------
    # Turn handling (intersection turns)
    # ------------------------------------------------------------------
    def _start_turn(self, direction, current_time):
        self.state = "TURNING"
        self.turn_start_time = current_time
        self.turn_duration = 0.7  # seconds for 90-degree turn
        if direction == "left":
            self.turn_left_speed = 0.2
            self.turn_right_speed = 0.8
        elif direction == "right":
            self.turn_left_speed = 0.8
            self.turn_right_speed = 0.2
        else:  # forward — just drive straight through
            self.turn_left_speed = self.base_speed
            self.turn_right_speed = self.base_speed
            self.turn_duration = 0.3

    def _handle_turn(self, current_time):
        elapsed = current_time - self.turn_start_time
        if elapsed < self.turn_duration:
            return self.turn_left_speed, self.turn_right_speed
        else:
            # Turn complete — advance grid and go back to following
            direction = None
            if self.turn_left_speed < self.turn_right_speed:
                direction = "left"
            elif self.turn_right_speed < self.turn_left_speed:
                direction = "right"
            else:
                direction = "forward"
            self._move_grid(direction)
            self.state = "FOLLOWING"
            return self.base_speed, self.base_speed

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
            # Spin in place: left backward, right forward (or vice versa)
            return 0.2, 0.8
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
            return self._pid_follow(cx)
        return self.base_speed, self.base_speed

    # ------------------------------------------------------------------
    # Recovery (lost line, spinning to find it)
    # ------------------------------------------------------------------
    def _handle_recovery(self):
        if self.last_known_error > 0:
            return 0.6, 0.0
        else:
            return 0.0, 0.6
