"""
lane_detector.py — Industry-level 10-stage lane detection pipeline.

Pipeline stages:
  1. Depth mask            — reject non-road pixels via RealSense depth
  2. Multi-colour-space    — HSV yellow + HSV white + LAB B-channel
  3. Morphological cleanup — open → close
  4. ROI mask              — trapezoidal region of interest
  5. Bird's-eye view       — perspective warp
  6. Sliding window search — find lane pixels in BEV
  7. Polynomial fit        — 2nd-degree curve per lane
  8. Temporal smoothing    — EMA on polynomial coefficients
  9. Curvature & offset    — radius of curvature, lateral offset
 10. Inverse warp          — project lane polygon back to camera view
"""

import cv2
import numpy as np

from config import (
    CAMERA_WIDTH, CAMERA_HEIGHT,
    # colour
    YELLOW_HSV_LOW_1, YELLOW_HSV_HIGH_1,
    YELLOW_HSV_LOW_2, YELLOW_HSV_HIGH_2,
    WHITE_HSV_LOW, WHITE_HSV_HIGH,
    GLARE_HSV_LOW, GLARE_HSV_HIGH,
    LAB_USE_OTSU, MORPH_KERNEL_SIZE,
    # ROI
    ROI_VERTICES_NORM,
    # BEV
    BEV_SRC_NORM, BEV_DST_NORM, BEV_WIDTH, BEV_HEIGHT,
    # sliding window
    SW_NUM_WINDOWS, SW_MARGIN_PX, SW_MIN_PIX, SW_HIST_SMOOTH_K,
    # poly
    POLY_ORDER, EMA_ALPHA, MIN_LANE_PIXELS,
    YM_PER_PIX, XM_PER_PIX,
)


class LaneDetector:
    """
    Detects left and right lane boundaries, fits polynomial curves, and
    provides curvature / offset data for the controller.
    """

    def __init__(self):
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
        )

        # BEV matrices (computed once for a fixed resolution)
        self._M_bev = None
        self._M_bev_inv = None
        self._bev_computed_for = None

        # temporal smoothing state
        self._left_fit  = None   # np.array([a, b, c])
        self._right_fit = None
        self._left_fit_history  = []
        self._right_fit_history = []

        # sliding window debug image (exposed for visualiser)
        self.bev_debug = None

        # last result cache
        self.left_fit_px   = None
        self.right_fit_px  = None
        self.left_curv_m   = None
        self.right_curv_m  = None
        self.center_offset_m = 0.0
        self.confidence    = 0.0    # 0–1

        self._no_lane_count = 0

    # ══════════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ══════════════════════════════════════════════════════════════════
    def process(self, rgb, depth=None):
        """
        Run the full 10-stage pipeline.

        Returns
        -------
        overlay : np.ndarray  (same shape as rgb)
            Colour image with filled lane polygon + boundary curves.
        """
        h, w = rgb.shape[:2]

        # 1 ── depth mask ────────────────────────────────────────────
        depth_mask = self._depth_mask(depth, h, w)

        # 2 ── multi-colour-space thresholding ───────────────────────
        lane_mask = self._colour_threshold(rgb)

        # apply depth mask
        if depth_mask is not None:
            lane_mask = cv2.bitwise_and(lane_mask, depth_mask)

        # 3 ── morphological cleanup ─────────────────────────────────
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN,  self._kernel, iterations=1)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, self._kernel, iterations=1)

        # 4 ── ROI mask ──────────────────────────────────────────────
        lane_mask = self._apply_roi(lane_mask, h, w)

        # 5 ── bird's-eye view ───────────────────────────────────────
        bev_mask = self._to_bev(lane_mask, h, w)

        # 6 ── sliding window search ─────────────────────────────────
        left_x, left_y, right_x, right_y = self._sliding_window(bev_mask)

        # 7 ── polynomial fit ────────────────────────────────────────
        left_fit, right_fit = self._fit_polynomials(
            left_x, left_y, right_x, right_y
        )

        # 8 ── temporal smoothing ────────────────────────────────────
        left_fit  = self._smooth(left_fit,  "left")
        right_fit = self._smooth(right_fit, "right")

        self.left_fit_px  = left_fit
        self.right_fit_px = right_fit

        # 9 ── curvature & offset ────────────────────────────────────
        self._compute_curvature_offset(left_fit, right_fit)

        # 10 ── inverse warp overlay ─────────────────────────────────
        overlay = self._draw_lane_overlay(rgb, left_fit, right_fit, h, w)

        return overlay

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 1 — DEPTH MASK
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def _depth_mask(depth, h, w):
        if depth is None:
            return None
        # resize depth to match RGB if needed
        dh, dw = depth.shape[:2]
        if (dh, dw) != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        from config import DEPTH_MIN_M, DEPTH_MAX_M
        valid = np.isfinite(depth) & (depth > DEPTH_MIN_M) & (depth < DEPTH_MAX_M)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[valid] = 255
        return mask

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 2 — MULTI-COLOUR-SPACE THRESHOLDING
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def _colour_threshold(rgb):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)

        # yellow (two ranges)
        y1 = cv2.inRange(hsv, YELLOW_HSV_LOW_1, YELLOW_HSV_HIGH_1)
        y2 = cv2.inRange(hsv, YELLOW_HSV_LOW_2, YELLOW_HSV_HIGH_2)
        yellow = cv2.bitwise_or(y1, y2)

        # white
        white = cv2.inRange(hsv, WHITE_HSV_LOW, WHITE_HSV_HIGH)

        # glare rejection
        glare = cv2.inRange(hsv, GLARE_HSV_LOW, GLARE_HSV_HIGH)
        white = cv2.bitwise_and(white, cv2.bitwise_not(glare))

        # LAB B-channel for yellow robustness
        if LAB_USE_OTSU:
            lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)
            _, b_bin = cv2.threshold(
                lab[:, :, 2], 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            yellow = cv2.bitwise_and(yellow, b_bin)

        # combine
        combined = cv2.bitwise_or(yellow, white)
        return combined

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 4 — ROI MASK
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def _apply_roi(mask, h, w):
        vertices = (ROI_VERTICES_NORM * np.array([w, h])).astype(np.int32)
        roi = np.zeros_like(mask)
        cv2.fillPoly(roi, [vertices], 255)
        return cv2.bitwise_and(mask, roi)

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 5 — BIRD'S-EYE VIEW
    # ══════════════════════════════════════════════════════════════════
    def _to_bev(self, mask, h, w):
        key = (h, w)
        if self._bev_computed_for != key:
            src = (BEV_SRC_NORM * np.array([w, h])).astype(np.float32)
            dst = (BEV_DST_NORM * np.array([BEV_WIDTH, BEV_HEIGHT])).astype(np.float32)
            self._M_bev     = cv2.getPerspectiveTransform(src, dst)
            self._M_bev_inv = cv2.getPerspectiveTransform(dst, src)
            self._bev_computed_for = key

        bev = cv2.warpPerspective(mask, self._M_bev, (BEV_WIDTH, BEV_HEIGHT))
        return bev

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 6 — SLIDING WINDOW SEARCH
    # ══════════════════════════════════════════════════════════════════
    def _sliding_window(self, bev):
        """Find left and right lane pixel coordinates in BEV image."""
        h, w = bev.shape[:2]

        # histogram of bottom half
        histogram = np.sum(bev[h // 2:, :], axis=0).astype(np.float64)
        if SW_HIST_SMOOTH_K > 1:
            kernel = np.ones(SW_HIST_SMOOTH_K) / SW_HIST_SMOOTH_K
            histogram = np.convolve(histogram, kernel, mode="same")

        midpoint  = w // 2
        left_base  = int(np.argmax(histogram[:midpoint]))
        right_base = int(np.argmax(histogram[midpoint:])) + midpoint

        window_h = h // SW_NUM_WINDOWS
        nonzero  = bev.nonzero()
        nz_y     = np.array(nonzero[0])
        nz_x     = np.array(nonzero[1])

        left_current  = left_base
        right_current = right_base

        left_inds  = []
        right_inds = []

        # debug image
        self.bev_debug = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)

        for win in range(SW_NUM_WINDOWS):
            y_low  = h - (win + 1) * window_h
            y_high = h - win * window_h

            xl_lo = left_current  - SW_MARGIN_PX
            xl_hi = left_current  + SW_MARGIN_PX
            xr_lo = right_current - SW_MARGIN_PX
            xr_hi = right_current + SW_MARGIN_PX

            # draw windows on debug
            cv2.rectangle(self.bev_debug, (xl_lo, y_low), (xl_hi, y_high), (0, 255, 0), 2)
            cv2.rectangle(self.bev_debug, (xr_lo, y_low), (xr_hi, y_high), (0, 0, 255), 2)

            good_left  = ((nz_y >= y_low) & (nz_y < y_high) &
                          (nz_x >= xl_lo) & (nz_x < xl_hi)).nonzero()[0]
            good_right = ((nz_y >= y_low) & (nz_y < y_high) &
                          (nz_x >= xr_lo) & (nz_x < xr_hi)).nonzero()[0]

            left_inds.append(good_left)
            right_inds.append(good_right)

            if len(good_left)  > SW_MIN_PIX:
                left_current  = int(np.mean(nz_x[good_left]))
            if len(good_right) > SW_MIN_PIX:
                right_current = int(np.mean(nz_x[good_right]))

        left_inds  = np.concatenate(left_inds)  if left_inds  else np.array([], dtype=int)
        right_inds = np.concatenate(right_inds) if right_inds else np.array([], dtype=int)

        left_x  = nz_x[left_inds]  if len(left_inds)  else np.array([])
        left_y  = nz_y[left_inds]  if len(left_inds)  else np.array([])
        right_x = nz_x[right_inds] if len(right_inds) else np.array([])
        right_y = nz_y[right_inds] if len(right_inds) else np.array([])

        return left_x, left_y, right_x, right_y

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 7 — POLYNOMIAL FIT
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def _fit_polynomials(lx, ly, rx, ry):
        left_fit  = None
        right_fit = None

        if len(ly) >= MIN_LANE_PIXELS:
            try:
                left_fit = np.polyfit(ly, lx, POLY_ORDER)
            except np.RankWarning:
                pass

        if len(ry) >= MIN_LANE_PIXELS:
            try:
                right_fit = np.polyfit(ry, rx, POLY_ORDER)
            except np.RankWarning:
                pass

        return left_fit, right_fit

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 8 — TEMPORAL SMOOTHING  (EMA)
    # ══════════════════════════════════════════════════════════════════
    def _smooth(self, new_fit, side):
        if side == "left":
            prev = self._left_fit
        else:
            prev = self._right_fit

        if new_fit is None:
            return prev            # hold last good fit

        if prev is None:
            smoothed = new_fit
        else:
            smoothed = EMA_ALPHA * new_fit + (1 - EMA_ALPHA) * prev

        if side == "left":
            self._left_fit = smoothed
        else:
            self._right_fit = smoothed

        return smoothed

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 9 — CURVATURE & LATERAL OFFSET
    # ══════════════════════════════════════════════════════════════════
    def _compute_curvature_offset(self, left_fit, right_fit):
        y_eval = BEV_HEIGHT - 1

        # -- curvature (in metres) --
        if left_fit is not None:
            a, b, _ = left_fit
            a_m = a * XM_PER_PIX / (YM_PER_PIX ** 2)
            b_m = b * XM_PER_PIX / YM_PER_PIX
            denom = (1 + (2 * a_m * y_eval * YM_PER_PIX + b_m) ** 2) ** 1.5
            self.left_curv_m = abs(denom / (2 * a_m + 1e-9))
        else:
            self.left_curv_m = None

        if right_fit is not None:
            a, b, _ = right_fit
            a_m = a * XM_PER_PIX / (YM_PER_PIX ** 2)
            b_m = b * XM_PER_PIX / YM_PER_PIX
            denom = (1 + (2 * a_m * y_eval * YM_PER_PIX + b_m) ** 2) ** 1.5
            self.right_curv_m = abs(denom / (2 * a_m + 1e-9))
        else:
            self.right_curv_m = None

        # -- lateral offset from lane centre --
        lane_center = BEV_WIDTH / 2.0
        if left_fit is not None and right_fit is not None:
            left_x  = np.polyval(left_fit,  y_eval)
            right_x = np.polyval(right_fit, y_eval)
            detected_center = (left_x + right_x) / 2.0
            self.center_offset_m = (detected_center - lane_center) * XM_PER_PIX
            self.confidence = 1.0
            self._no_lane_count = 0
        elif left_fit is not None:
            left_x = np.polyval(left_fit, y_eval)
            # estimate centre assuming fixed lane width (~0.35 m in BEV pixels)
            est_width = 0.35 / XM_PER_PIX
            detected_center = left_x + est_width / 2.0
            self.center_offset_m = (detected_center - lane_center) * XM_PER_PIX
            self.confidence = 0.5
            self._no_lane_count = 0
        elif right_fit is not None:
            right_x = np.polyval(right_fit, y_eval)
            est_width = 0.35 / XM_PER_PIX
            detected_center = right_x - est_width / 2.0
            self.center_offset_m = (detected_center - lane_center) * XM_PER_PIX
            self.confidence = 0.5
            self._no_lane_count = 0
        else:
            self.confidence = 0.0
            self._no_lane_count += 1

    @property
    def no_lane_count(self):
        return self._no_lane_count

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 10 — INVERSE WARP OVERLAY
    # ══════════════════════════════════════════════════════════════════
    def _draw_lane_overlay(self, rgb, left_fit, right_fit, h, w):
        overlay = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)
        plot_y = np.linspace(0, BEV_HEIGHT - 1, BEV_HEIGHT).astype(int)

        if left_fit is not None and right_fit is not None:
            left_x  = np.polyval(left_fit,  plot_y).astype(int)
            right_x = np.polyval(right_fit, plot_y).astype(int)

            # clamp to image bounds
            left_x  = np.clip(left_x,  0, BEV_WIDTH - 1)
            right_x = np.clip(right_x, 0, BEV_WIDTH - 1)

            # fill polygon
            pts_left  = np.column_stack((left_x,  plot_y))
            pts_right = np.column_stack((right_x, plot_y))[::-1]
            pts = np.vstack((pts_left, pts_right))

            from config import OVERLAY_LANE_COLOR
            cv2.fillPoly(overlay, [pts], OVERLAY_LANE_COLOR)

            # draw lane boundaries
            from config import OVERLAY_LEFT_COLOR, OVERLAY_RIGHT_COLOR, OVERLAY_THICKNESS
            for i in range(len(plot_y) - 1):
                cv2.line(overlay,
                         (left_x[i], plot_y[i]), (left_x[i+1], plot_y[i+1]),
                         OVERLAY_LEFT_COLOR, OVERLAY_THICKNESS)
                cv2.line(overlay,
                         (right_x[i], plot_y[i]), (right_x[i+1], plot_y[i+1]),
                         OVERLAY_RIGHT_COLOR, OVERLAY_THICKNESS)

            # draw centre line
            center_x = ((left_x + right_x) // 2).astype(int)
            from config import OVERLAY_CENTER_COLOR
            for i in range(len(plot_y) - 1):
                cv2.line(overlay,
                         (center_x[i], plot_y[i]), (center_x[i+1], plot_y[i+1]),
                         OVERLAY_CENTER_COLOR, 2)

        elif left_fit is not None:
            left_x = np.clip(np.polyval(left_fit, plot_y).astype(int), 0, BEV_WIDTH - 1)
            from config import OVERLAY_LEFT_COLOR, OVERLAY_THICKNESS
            for i in range(len(plot_y) - 1):
                cv2.line(overlay,
                         (left_x[i], plot_y[i]), (left_x[i+1], plot_y[i+1]),
                         OVERLAY_LEFT_COLOR, OVERLAY_THICKNESS)

        elif right_fit is not None:
            right_x = np.clip(np.polyval(right_fit, plot_y).astype(int), 0, BEV_WIDTH - 1)
            from config import OVERLAY_RIGHT_COLOR, OVERLAY_THICKNESS
            for i in range(len(plot_y) - 1):
                cv2.line(overlay,
                         (right_x[i], plot_y[i]), (right_x[i+1], plot_y[i+1]),
                         OVERLAY_RIGHT_COLOR, OVERLAY_THICKNESS)

        # inverse warp back to camera perspective
        if self._M_bev_inv is not None:
            unwarped = cv2.warpPerspective(overlay, self._M_bev_inv, (w, h))
        else:
            unwarped = cv2.resize(overlay, (w, h))

        # blend with original
        result = cv2.addWeighted(rgb, 1.0, unwarped, 0.6, 0)
        return result
