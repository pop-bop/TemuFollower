from config import RK4_HEADING_GAIN, RK4_MAX_DT, RK4_TRAJECTORY_GAIN
from utils import clamp


def new_trajectory_state():
    return {"error": 0.0, "heading": 0.0}


def _derivatives(state, target_error, target_heading):
    return {
        "error": RK4_TRAJECTORY_GAIN * (target_error - state["error"]),
        "heading": RK4_HEADING_GAIN * (target_heading - state["heading"]),
    }


def _advance(state, derivs, dt_scale):
    return {
        "error": state["error"] + derivs["error"] * dt_scale,
        "heading": state["heading"] + derivs["heading"] * dt_scale,
    }


def rk4_step(state, target_error, target_heading, dt):
    dt = clamp(dt, 0.001, RK4_MAX_DT)

    k1 = _derivatives(state, target_error, target_heading)
    k2 = _derivatives(_advance(state, k1, dt * 0.5), target_error, target_heading)
    k3 = _derivatives(_advance(state, k2, dt * 0.5), target_error, target_heading)
    k4 = _derivatives(_advance(state, k3, dt), target_error, target_heading)

    state["error"] += (dt / 6.0) * (
        k1["error"] + 2.0 * k2["error"] + 2.0 * k3["error"] + k4["error"]
    )
    state["heading"] += (dt / 6.0) * (
        k1["heading"] + 2.0 * k2["heading"] + 2.0 * k3["heading"] + k4["heading"]
    )

    state["error"] = clamp(state["error"], -1.0, 1.0)
    state["heading"] = clamp(state["heading"], -1.0, 1.0)
    return state["error"], state["heading"]


def reset_trajectory_state(state, error=0.0, heading=0.0):
    state["error"] = clamp(error, -1.0, 1.0)
    state["heading"] = clamp(heading, -1.0, 1.0)
