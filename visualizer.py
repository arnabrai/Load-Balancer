"""
visualizer.py — Rich HUD overlay, lane fill, and BEV minimap for QCar 2.

Draws on top of the already-lane-overlaid frame produced by LaneDetector.
Adds:
  • semi-transparent dark HUD panel with live telemetry
  • curvature direction arrow
  • bird's-eye-view minimap inset
  • colour-coded confidence bar
"""

import cv2
import numpy as np

from config import (
    HUD_FONT, HUD_FONT_SCALE, HUD_THICKNESS, HUD_LINE_HEIGHT,
    HUD_X, HUD_Y, HUD_BG_ALPHA,
    MINIMAP_SIZE, MINIMAP_POSITION,
    WINDOW_NAME, WINDOW_SIZE,
)


class Visualizer:
    """Composites all visual overlays onto the display frame."""

    def __init__(self):
        self._window_created = False

    # ══════════════════════════════════════════════════════════════════
    #  PUBLIC
    # ══════════════════════════════════════════════════════════════════
    def render(self, frame, detector, controller, odom, fps, manual_mode=False):
        """
        Draw HUD + minimap on `frame` (mutates in-place for speed).
        Returns the annotated frame.
        """
        h, w = frame.shape[:2]

        # ── HUD telemetry panel ─────────────────────────────────────
        lines = self._build_hud_lines(detector, controller, odom, fps, manual_mode)
        self._draw_hud_panel(frame, lines, w)

        # ── confidence bar ──────────────────────────────────────────
        self._draw_confidence_bar(frame, detector.confidence, w, h)

        # ── curvature arrow ─────────────────────────────────────────
        self._draw_curvature_arrow(frame, controller.steer_cmd, w, h)

        # ── BEV minimap ─────────────────────────────────────────────
        if detector.bev_debug is not None:
            self._draw_minimap(frame, detector.bev_debug, w, h)

        return frame

    def show(self, frame):
        """Display frame in named window. Returns pressed key or -1."""
        if not self._window_created:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, *WINDOW_SIZE)
            self._window_created = True
        cv2.imshow(WINDOW_NAME, frame)
        return cv2.waitKey(1) & 0xFF

    def destroy(self):
        cv2.destroyAllWindows()

    # ══════════════════════════════════════════════════════════════════
    #  HUD
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def _build_hud_lines(det, ctrl, odom, fps, manual):
        mode = "MANUAL" if manual else ("E-STOP" if ctrl.emergency_stop else "AUTO")
        lines = [
            f"Mode: {mode}   FPS: {fps}",
            f"Speed cmd: {ctrl.speed_cmd:.3f}   Steer: {ctrl.steer_cmd:+.3f} rad  ({ctrl.steer_angle_deg:+.1f} deg)",
            f"Offset: {det.center_offset_m:+.4f} m   Confidence: {det.confidence:.0%}",
        ]

        # curvature
        lc = f"{det.left_curv_m:.1f}" if det.left_curv_m else "—"
        rc = f"{det.right_curv_m:.1f}" if det.right_curv_m else "—"
        lines.append(f"Curvature  L: {lc} m   R: {rc} m")

        # odometry
        if odom:
            lines.append(
                f"Speed: {odom.get('v_filtered', 0):.3f} m/s   "
                f"Dist: {odom.get('total_dist', 0):.3f} m"
            )

        lines.append("Keys: ESC=quit  G=manual  P=pause  C=cam-reset  S=screenshot")
        return lines

    @staticmethod
    def _draw_hud_panel(frame, lines, w):
        """Semi-transparent dark background + text."""
        n = len(lines)
        panel_h = HUD_Y + n * HUD_LINE_HEIGHT + 10
        panel_w = min(w - 10, 680)

        # dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (panel_w, panel_h), (15, 15, 15), cv2.FILLED)
        cv2.addWeighted(overlay, HUD_BG_ALPHA, frame, 1 - HUD_BG_ALPHA, 0, frame)

        # text
        for i, text in enumerate(lines):
            y = HUD_Y + i * HUD_LINE_HEIGHT
            # determine colour
            if "E-STOP" in text:
                colour = (0, 0, 255)
            elif "MANUAL" in text:
                colour = (0, 200, 255)
            elif i == 0:
                colour = (0, 255, 120)
            else:
                colour = (220, 220, 220)

            cv2.putText(frame, text, (HUD_X, y),
                        HUD_FONT, HUD_FONT_SCALE, colour, HUD_THICKNESS,
                        cv2.LINE_AA)

    # ══════════════════════════════════════════════════════════════════
    #  CONFIDENCE BAR
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def _draw_confidence_bar(frame, confidence, w, h):
        bar_w = 200
        bar_h = 14
        x0 = w - bar_w - 15
        y0 = 15

        # background
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (50, 50, 50), cv2.FILLED)

        # fill
        fill_w = int(bar_w * confidence)
        if confidence >= 0.8:
            colour = (0, 220, 0)
        elif confidence >= 0.4:
            colour = (0, 200, 255)
        else:
            colour = (0, 0, 255)
        cv2.rectangle(frame, (x0, y0), (x0 + fill_w, y0 + bar_h), colour, cv2.FILLED)

        # border
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (180, 180, 180), 1)

        # label
        cv2.putText(frame, f"Lane: {confidence:.0%}", (x0, y0 + bar_h + 18),
                    HUD_FONT, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # ══════════════════════════════════════════════════════════════════
    #  CURVATURE ARROW
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def _draw_curvature_arrow(frame, steer, w, h):
        cx = w // 2
        cy = h - 50
        length = 60
        angle_rad = -steer * 3.0  # exaggerate for visibility

        ex = int(cx + length * np.sin(angle_rad))
        ey = int(cy - length * np.cos(angle_rad))

        cv2.arrowedLine(frame, (cx, cy), (ex, ey), (0, 255, 255), 3, tipLength=0.35)
        cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)

    # ══════════════════════════════════════════════════════════════════
    #  BEV MINIMAP
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def _draw_minimap(frame, bev, w, h):
        mw, mh = MINIMAP_SIZE
        mini = cv2.resize(bev, (mw, mh), interpolation=cv2.INTER_AREA)

        # add border
        cv2.rectangle(mini, (0, 0), (mw - 1, mh - 1), (200, 200, 200), 2)

        # position
        if MINIMAP_POSITION == "bottom-right":
            x0 = w - mw - 10
            y0 = h - mh - 10
        elif MINIMAP_POSITION == "bottom-left":
            x0 = 10
            y0 = h - mh - 10
        else:
            x0 = w - mw - 10
            y0 = 10

        # overlay
        if y0 + mh <= h and x0 + mw <= w and y0 >= 0 and x0 >= 0:
            frame[y0:y0 + mh, x0:x0 + mw] = mini

        # label
        cv2.putText(frame, "BEV", (x0 + 5, y0 - 5),
                    HUD_FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
