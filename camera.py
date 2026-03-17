"""
camera.py — SafeCamera3D: resilient Intel RealSense wrapper for QCar 2.

Handles auto-recovery from dropped frames, stale streams, and hardware
hiccups without crashing the control loop.
"""

import time
import numpy as np

try:
    from Quanser.q_essential import Camera3D
except ImportError:
    Camera3D = None
    print("[SafeCamera3D] WARNING: Quanser SDK not found. Camera will return synthetic frames.")

from config import (
    CAMERA_MODE, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, CAMERA_DEVICE_ID,
    CAMERA_FAIL_RESET_THRESHOLD, CAMERA_MAX_NO_GOOD_SECS,
    DEPTH_MIN_M, DEPTH_MAX_M,
)


class SafeCamera3D:
    """
    Wraps Quanser Camera3D with:
      • automatic stream reset after N consecutive failures
      • timeout-based reset if no good frame arrives within T seconds
      • clean terminate / reinit cycle
      • depth range masking convenience method
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._cam = None
        self._consec_fail = 0
        self._last_good_ts = 0.0
        self._init_camera()

    # ── logging ──────────────────────────────────────────────────────
    def _log(self, *args):
        if self.verbose:
            print("[SafeCamera3D]", *args)

    # ── lifecycle ────────────────────────────────────────────────────
    def _safe_terminate(self, cam):
        """Best-effort shutdown — tries multiple SDK stop methods."""
        for method in ("terminate_RGB", "stop_RGB", "stop_rgb", "stop"):
            fn = getattr(cam, method, None)
            if fn:
                try:
                    fn()
                except Exception:
                    pass
        try:
            cam.terminate()
        except AttributeError as exc:
            if "video3d" not in str(exc):
                raise
        except Exception:
            pass

    def _init_camera(self):
        if self._cam is not None:
            try:
                self._safe_terminate(self._cam)
            except Exception:
                pass

        if Camera3D is None:
            self._log("SDK not available — running in synthetic mode.")
            self._cam = None
            self._consec_fail = 0
            self._last_good_ts = time.time()
            return

        self._log(f"Init Camera3D  mode={CAMERA_MODE}  "
                  f"{CAMERA_WIDTH}×{CAMERA_HEIGHT}@{CAMERA_FPS}  "
                  f"dev={CAMERA_DEVICE_ID}")

        self._cam = Camera3D(
            mode=CAMERA_MODE,
            frame_width_RGB=CAMERA_WIDTH,
            frame_height_RGB=CAMERA_HEIGHT,
            frame_rate_RGB=CAMERA_FPS,
            device_id=CAMERA_DEVICE_ID,
        )
        self._consec_fail = 0
        self._last_good_ts = 0.0

        # warm-up read
        try:
            if "RGB" in CAMERA_MODE:
                self._cam.read_RGB()
            rgb = getattr(self._cam, "image_buffer_RGB", None)
            if self._is_valid(rgb):
                self._last_good_ts = time.time()
                self._log("Warm-up OK.")
        except Exception:
            self._log("Warm-up read failed (will recover).")

    # ── validation ───────────────────────────────────────────────────
    @staticmethod
    def _is_valid(img):
        return (
            img is not None
            and hasattr(img, "shape")
            and len(img.shape) >= 2
            and img.shape[0] > 0
            and img.shape[1] > 0
        )

    def _needs_reset(self) -> bool:
        if self._consec_fail >= CAMERA_FAIL_RESET_THRESHOLD:
            return True
        if self._last_good_ts and (time.time() - self._last_good_ts) > CAMERA_MAX_NO_GOOD_SECS:
            return True
        return False

    # ── public API ───────────────────────────────────────────────────
    def read(self):
        """
        Returns (rgb, depth) numpy arrays or (None, None) on failure.
        Depth is in metres when available.
        """
        # ---- synthetic fallback when SDK is absent -----------------
        if Camera3D is None or self._cam is None:
            synth_rgb = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
            synth_depth = np.ones((CAMERA_HEIGHT, CAMERA_WIDTH), dtype=np.float32) * 0.5
            return synth_rgb, synth_depth

        # ---- real camera -------------------------------------------
        try:
            if "RGB" in CAMERA_MODE:
                self._cam.read_RGB()
            if "DEPTH" in CAMERA_MODE:
                self._cam.read_depth(dataMode="m")

            rgb   = getattr(self._cam, "image_buffer_RGB", None)
            depth = getattr(self._cam, "image_buffer_depth_m",
                            getattr(self._cam, "image_buffer_depth", None))

            if self._is_valid(rgb):
                self._consec_fail = 0
                self._last_good_ts = time.time()
                return (
                    rgb.copy(),
                    depth.copy() if self._is_valid(depth) else None,
                )
            else:
                self._consec_fail += 1
                if self._needs_reset():
                    self._log("Invalid frames — resetting stream …")
                    self._init_camera()

        except Exception as exc:
            self._consec_fail += 1
            self._log(f"read() exception: {exc}")
            if self._needs_reset():
                self._log("Exceptions persisted — resetting stream …")
                self._init_camera()

        return None, None

    def force_reset(self):
        """Manual reset triggered by user hotkey."""
        self._log("Force reset.")
        self._init_camera()

    def terminate(self):
        if self._cam is not None:
            try:
                self._safe_terminate(self._cam)
            except Exception:
                pass
            self._cam = None
            self._log("Camera terminated.")

    # ── depth utilities ──────────────────────────────────────────────
    @staticmethod
    def create_depth_mask(depth, min_m=DEPTH_MIN_M, max_m=DEPTH_MAX_M):
        """
        Binary mask where 255 = pixel depth is within [min_m, max_m].
        Pixels with NaN / zero / out-of-range depth are rejected.
        """
        if depth is None:
            return None
        valid = np.isfinite(depth) & (depth > min_m) & (depth < max_m)
        mask = np.zeros(depth.shape[:2], dtype=np.uint8)
        mask[valid] = 255
        return mask
