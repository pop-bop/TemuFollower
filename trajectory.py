from config import (
    RK4_HEADING_GAIN, RK4_MAX_DT, RK4_TRAJECTORY_GAIN,
    RK4_CURVATURE_GAIN, RK4_CURVATURE_RATE_GAIN,
)
from utils import clamp


def new_trajectory_state():
    return {"error": 0.0, "heading": 0.0}


def _derivatives(state, target_error, target_heading, error_gain, heading_gain):
    return {
        "error": error_gain * (target_error - state["error"]),
        "heading": heading_gain * (target_heading - state["heading"]),
    }


def _advance(state, derivs, dt_scale):
    return {
        "error": state["error"] + derivs["error"] * dt_scale,
        "heading": state["heading"] + derivs["heading"] * dt_scale,
    }


def _rk4_integrate(state, target_error, target_heading, dt, error_gain, heading_gain):
    dt = clamp(dt, 0.001, RK4_MAX_DT)

    k1 = _derivatives(state, target_error, target_heading, error_gain, heading_gain)
    k2 = _derivatives(_advance(state, k1, dt * 0.5), target_error, target_heading, error_gain, heading_gain)
    k3 = _derivatives(_advance(state, k2, dt * 0.5), target_error, target_heading, error_gain, heading_gain)
    k4 = _derivatives(_advance(state, k3, dt), target_error, target_heading, error_gain, heading_gain)

    state["error"] += (dt / 6.0) * (
        k1["error"] + 2.0 * k2["error"] + 2.0 * k3["error"] + k4["error"]
    )
    state["heading"] += (dt / 6.0) * (
        k1["heading"] + 2.0 * k2["heading"] + 2.0 * k3["heading"] + k4["heading"]
    )

    state["error"] = clamp(state["error"], -1.0, 1.0)
    state["heading"] = clamp(state["heading"], -1.0, 1.0)
    return state["error"], state["heading"]


def rk4_step(state, target_error, target_heading, dt):
    """Layer 1: near/lookahead ROIs -> predicted error + heading (curve estimate)."""
    return _rk4_integrate(state, target_error, target_heading, dt, RK4_TRAJECTORY_GAIN, RK4_HEADING_GAIN)


def rk4_step_curvature(state, target_curvature, target_curvature_rate, dt):
    """Layer 2: lookahead/far ROIs -> predicted curvature + curvature rate. Lets
    the robot anticipate a turn closing out before the near ROI ever straightens."""
    return _rk4_integrate(state, target_curvature, target_curvature_rate, dt, RK4_CURVATURE_GAIN, RK4_CURVATURE_RATE_GAIN)


def reset_trajectory_state(state, error=0.0, heading=0.0):
    state["error"] = clamp(error, -1.0, 1.0)
    state["heading"] = clamp(heading, -1.0, 1.0)
