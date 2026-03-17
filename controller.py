"""
controller.py — PID steering + adaptive speed controller for QCar 2.

Steering : PID on lateral offset from lane centre (pixels → radians).
Speed    : Base speed reduced proportional to curve tightness.
Safety   : Emergency stop after N frames with no lane detected.
"""

import time
import numpy as np

from config import (
    PID_KP, PID_KI, PID_KD, PID_INTEGRAL_MAX,
    STEER_MAX_RAD, STEER_DEADBAND,
    SPEED_BASE, SPEED_MIN, SPEED_MAX, SPEED_CURVE_GAIN, SPEED_TURN_COS_MIN,
    NO_LANE_STOP_FRAMES, THROTTLE_SATURATION,
    LEDS_DEFAULT,
)


class PIDController:
    """Discrete PID with anti-windup and derivative filtering."""

    def __init__(self, kp=PID_KP, ki=PID_KI, kd=PID_KD):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._integral  = 0.0
        self._prev_err  = 0.0
        self._prev_time = None

    def reset(self):
        self._integral  = 0.0
        self._prev_err  = 0.0
        self._prev_time = None

    def update(self, error: float) -> float:
        now = time.time()
        if self._prev_time is None:
            dt = 0.033  # assume ~30 fps on first call
        else:
            dt = max(now - self._prev_time, 1e-6)
        self._prev_time = now

        # proportional
        p = self.kp * error

        # integral with anti-windup clamp
        self._integral += error * dt
        self._integral  = np.clip(self._integral, -PID_INTEGRAL_MAX, PID_INTEGRAL_MAX)
        i = self.ki * self._integral

        # derivative (on error, not on measurement — simple version)
        d = self.kd * (error - self._prev_err) / dt
        self._prev_err = error

        return p + i + d


class LaneController:
    """
    Converts lane-detector output into motor commands.

    Call `compute(detector)` each frame to get `(speed, steer, leds)`.
    """

    def __init__(self):
        self.pid = PIDController()

        # public state (for HUD)
        self.steer_cmd = 0.0
        self.speed_cmd = SPEED_BASE
        self.steer_angle_deg = 0.0
        self.emergency_stop = False

    def compute(self, detector):
        """
        Parameters
        ----------
        detector : LaneDetector
            Must have been `.process()`'d this frame.

        Returns
        -------
        mtr_cmd : np.ndarray [speed, steer]
        leds    : np.ndarray [8]
        """
        offset_px = detector.center_offset_m / 0.008  # back to px-ish units
        confidence = detector.confidence
        no_lane    = detector.no_lane_count

        # ── emergency stop ──────────────────────────────────────────
        if no_lane >= NO_LANE_STOP_FRAMES:
            self.emergency_stop = True
            self.steer_cmd = 0.0
            self.speed_cmd = 0.0
            self.steer_angle_deg = 0.0
            return self._make_cmd()

        self.emergency_stop = False

        # ── steering (PID on lateral offset) ────────────────────────
        if confidence > 0:
            if abs(offset_px) < STEER_DEADBAND:
                raw_steer = 0.0
                self.pid._prev_err = 0.0   # don't accumulate integral in deadband
            else:
                raw_steer = self.pid.update(offset_px)

            self.steer_cmd = float(np.clip(raw_steer, -STEER_MAX_RAD, STEER_MAX_RAD))
        else:
            # no lane but under threshold — hold last steer and slow down
            self.speed_cmd = max(SPEED_MIN, self.speed_cmd * 0.95)

        self.steer_angle_deg = np.degrees(self.steer_cmd)

        # ── adaptive speed ──────────────────────────────────────────
        curvatures = []
        if detector.left_curv_m is not None:
            curvatures.append(detector.left_curv_m)
        if detector.right_curv_m is not None:
            curvatures.append(detector.right_curv_m)

        if curvatures:
            avg_curv = np.mean(curvatures)
            # reduce speed for tight curves
            curve_penalty = SPEED_CURVE_GAIN / max(avg_curv, 0.1)
            base = SPEED_BASE - curve_penalty
        else:
            base = SPEED_BASE

        # cos-based turn slow-down (from q_dp.py pattern)
        cos_gain = max(SPEED_TURN_COS_MIN, np.cos(abs(self.steer_cmd)))
        self.speed_cmd = float(np.clip(base * cos_gain, SPEED_MIN, SPEED_MAX))

        # confidence scaling — reduce speed when only one lane visible
        if confidence < 1.0:
            self.speed_cmd *= (0.6 + 0.4 * confidence)
            self.speed_cmd = max(SPEED_MIN, self.speed_cmd)

        return self._make_cmd()

    def _make_cmd(self):
        speed = float(np.clip(self.speed_cmd, -THROTTLE_SATURATION, THROTTLE_SATURATION))
        steer = float(np.clip(self.steer_cmd, -STEER_MAX_RAD, STEER_MAX_RAD))

        mtr_cmd = np.array([speed, steer], dtype=np.float64)

        # LED indicators
        leds = LEDS_DEFAULT.copy()
        if steer > 0.05:        # turning left
            leds[0] = 1; leds[2] = 1
        elif steer < -0.05:     # turning right
            leds[1] = 1; leds[3] = 1
        if self.emergency_stop:
            leds[4] = 1; leds[5] = 1  # hazard

        return mtr_cmd, leds

    def stop(self):
        """Immediate zero command."""
        self.speed_cmd = 0.0
        self.steer_cmd = 0.0
        return self._make_cmd()
