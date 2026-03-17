"""
odometry.py — EMA-filtered encoder odometry for QCar 2.

Reads the motor encoder, computes:
  • instantaneous speed (m/s) with exponential moving average
  • cumulative distance travelled (m)

Robust to encoder read failures and timing outliers.
"""

import time
import numpy as np

from config import (
    ENCODER_TICKS_PER_REV, WHEEL_CIRCUMFERENCE_M,
    ODOM_EMA_ALPHA, ODOM_DT_MIN, ODOM_DT_MAX, ODOM_V_HARD_MAX,
)


class EncoderOdometry:
    """Wraps QCar encoder reads with filtering and cumulative tracking."""

    def __init__(self):
        self.v_filtered   = 0.0      # EMA speed (m/s)
        self.v_raw        = 0.0      # unfiltered speed
        self.total_dist_m = 0.0      # cumulative distance
        self.d_dist_m     = 0.0      # distance since last update

        self._prev_ticks  = None
        self._prev_t      = None
        self._start_ticks = None

    # ── lifecycle ────────────────────────────────────────────────────
    def reset(self, qcar):
        """Reset baselines. Call before a new run segment."""
        ticks = self._read_ticks(qcar)
        now   = time.time()

        self._prev_ticks  = ticks
        self._prev_t      = now
        self._start_ticks = ticks
        self.v_filtered   = 0.0
        self.v_raw        = 0.0
        self.total_dist_m = 0.0
        self.d_dist_m     = 0.0

    # ── public API ───────────────────────────────────────────────────
    def update(self, qcar) -> dict:
        """
        Read encoder, compute speed and distance.

        Returns dict with keys:
            ticks, d_ticks, d_dist, total_dist, v_raw, v_filtered, dt
        """
        now   = time.time()
        ticks = self._read_ticks(qcar)

        if self._prev_t is None or self._prev_ticks is None:
            self._prev_ticks  = ticks
            self._prev_t      = now
            self._start_ticks = ticks
            return self._zero_result(ticks)

        dt = max(1e-6, now - self._prev_t)
        d_ticks = ticks - self._prev_ticks

        self._prev_ticks = ticks
        self._prev_t     = now

        # distance from ticks
        d_dist = (d_ticks / ENCODER_TICKS_PER_REV) * WHEEL_CIRCUMFERENCE_M
        v_raw  = d_dist / dt

        # outlier rejection
        if dt < ODOM_DT_MIN or dt > ODOM_DT_MAX or abs(v_raw) > ODOM_V_HARD_MAX:
            v_raw = self.v_filtered  # hold previous

        # EMA
        self.v_filtered = ODOM_EMA_ALPHA * v_raw + (1 - ODOM_EMA_ALPHA) * self.v_filtered
        self.v_raw      = v_raw
        self.d_dist_m   = d_dist
        self.total_dist_m += abs(d_dist)

        return {
            "ticks":      ticks,
            "d_ticks":    d_ticks,
            "d_dist":     d_dist,
            "total_dist": self.total_dist_m,
            "v_raw":      v_raw,
            "v_filtered": self.v_filtered,
            "dt":         dt,
        }

    # ── internals ────────────────────────────────────────────────────
    @staticmethod
    def _read_ticks(qcar) -> float:
        try:
            val = qcar.read_encoder()
            if isinstance(val, (list, tuple, np.ndarray)):
                return float(val[0])
            return float(val)
        except Exception:
            return 0.0

    @staticmethod
    def _zero_result(ticks):
        return {
            "ticks":      ticks,
            "d_ticks":    0.0,
            "d_dist":     0.0,
            "total_dist": 0.0,
            "v_raw":      0.0,
            "v_filtered": 0.0,
            "dt":         0.0,
        }
