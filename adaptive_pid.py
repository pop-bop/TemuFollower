from config import (
    ADAPTIVE_PID_ENABLED, ADAPTIVE_KP_ERROR_BOOST, ADAPTIVE_KP_CONFIDENCE_DROP,
    ADAPTIVE_KD_ERROR_BOOST, ADAPTIVE_KD_DERIVATIVE_BOOST,
    ADAPTIVE_KI_ERROR_REDUCTION, ADAPTIVE_DERIVATIVE_REF,
)
from utils import clamp


def schedule_pid_gains(base_kp, base_ki, base_kd, error, derivative, line_confidence):
    if not ADAPTIVE_PID_ENABLED:
        return base_kp, base_ki, base_kd

    abs_error = abs(error)
    derivative_load = clamp(abs(derivative) / max(0.001, ADAPTIVE_DERIVATIVE_REF), 0.0, 1.0)
    confidence = clamp(line_confidence, 0.0, 1.0)
    confidence_loss = 1.0 - confidence

    kp = base_kp * (
        1.0
        + ADAPTIVE_KP_ERROR_BOOST * abs_error
        - ADAPTIVE_KP_CONFIDENCE_DROP * confidence_loss
    )
    ki = base_ki * max(0.0, 1.0 - ADAPTIVE_KI_ERROR_REDUCTION * abs_error)
    kd = base_kd * (
        1.0
        + ADAPTIVE_KD_ERROR_BOOST * abs_error
        + ADAPTIVE_KD_DERIVATIVE_BOOST * derivative_load
    )

    return max(0.0, kp), max(0.0, ki), max(0.0, kd)
