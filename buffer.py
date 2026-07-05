import time

from config import INTERSECTION_MEMORY_MAX_AGE_S, MAX_WAYPOINTS


class TemporalBuffer:
    def __init__(self, max_size=MAX_WAYPOINTS):
        self.waypoints = []
        self.max_size = max_size

    def record(self, frame=None, state=None, error=None, debug_info=None,
               is_intersection=False, branches=None, chosen_branch_idx=None):
        branch_memory = []
        if branches:
            branch_memory = [
                {
                    "cx": branch.get("cx"),
                    "cy": branch.get("cy"),
                    "taken": branch.get("taken", False),
                }
                for branch in branches
            ]

        wp = {
            "timestamp": time.perf_counter(),
            "state": state,
            "error": error,
            "is_intersection": is_intersection,
            "branches": branch_memory,
            "chosen_branch_idx": chosen_branch_idx,
            "dead_end": False,
            "frame": frame.copy() if frame is not None else None,
        }
        self.waypoints.append(wp)
        if len(self.waypoints) > self.max_size:
            self.waypoints.pop(0)
        return wp

    def record_dead_end(self, frame=None, state=None, error=None):
        wp = self.record(frame=frame, state=state, error=error)
        wp["dead_end"] = True
        return wp

    def peek(self):
        return self.waypoints[-1] if self.waypoints else None

    def find_recent_intersection(self, now, max_age=INTERSECTION_MEMORY_MAX_AGE_S):
        for wp in reversed(self.waypoints):
            if (wp.get("is_intersection") and wp.get("branches")
                    and now - wp["timestamp"] < max_age):
                return wp
        return None

    def find_untried_branch(self, intersection_wp):
        if intersection_wp is None:
            return None, None
        chosen = intersection_wp.get("chosen_branch_idx")
        branches = intersection_wp.get("branches")
        if not branches:
            return None, None
        for i, branch in enumerate(branches):
            if i != chosen and not branch.get("taken", False):
                return i, branch
        return None, None

    def mark_branch_taken(self, intersection_wp, branch_idx):
        if (intersection_wp and intersection_wp.get("branches")
                and 0 <= branch_idx < len(intersection_wp["branches"])):
            intersection_wp["branches"][branch_idx]["taken"] = True

    def mark_chosen_branch_taken(self, intersection_wp):
        if intersection_wp is None:
            return
        chosen = intersection_wp.get("chosen_branch_idx")
        if chosen is not None:
            self.mark_branch_taken(intersection_wp, chosen)

    def has_untried_branches(self, intersection_wp):
        if intersection_wp is None:
            return False
        _, branch = self.find_untried_branch(intersection_wp)
        return branch is not None

    def clear_after(self, timestamp):
        self.waypoints = [wp for wp in self.waypoints if wp["timestamp"] <= timestamp]
