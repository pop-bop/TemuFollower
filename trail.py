from collections import deque


def new_trail():
    return deque()


def record_sample(trail, forward, turn, dt, now, max_age):
    trail.append({"forward": forward, "turn": turn, "dt": dt, "timestamp": now})
    while trail and now - trail[0]["timestamp"] > max_age:
        trail.popleft()


def pop_retrace_step(trail):
    return trail.pop() if trail else None
