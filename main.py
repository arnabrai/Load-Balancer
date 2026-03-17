#!/usr/bin/env python3
"""
main.py — QCar 2 Industry-Level Lane Detection & Following System.

Usage:
    python3 main.py                  # full autonomous mode
    python3 main.py --preview        # camera + detection only (no driving)
    python3 main.py --test-image X   # run pipeline on a static image

Controls:
    ESC   quit
    G     toggle manual gamepad mode
    P     pause / resume
    C     force camera reset
    S     save screenshot
"""

import sys
import os
import time
import argparse
import datetime
import numpy as np
import cv2

# ── project modules ─────────────────────────────────────────────────
from config import LEDS_DEFAULT, CAMERA_FPS
from camera import SafeCamera3D
from lane_detector import LaneDetector
from controller import LaneController
from odometry import EncoderOdometry
from visualizer import Visualizer

# ── optional Quanser imports ────────────────────────────────────────
try:
    from Quanser.product_QCar import QCar
    HAS_QCAR = True
except ImportError:
    HAS_QCAR = False
    print("[main] WARNING: Quanser QCar SDK not found — running in preview-only mode.")

try:
    from Quanser.q_ui import gamepadViaTarget
    HAS_GAMEPAD = True
except ImportError:
    HAS_GAMEPAD = False


# ════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════

def neutral(car):
    """Send zero command to motors."""
    try:
        car.read_write_std(
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
        )
    except Exception:
        pass


def save_screenshot(frame, directory="screenshots"):
    os.makedirs(directory, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(directory, f"lane_{ts}.png")
    cv2.imwrite(path, frame)
    print(f"[Screenshot] Saved → {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="QCar 2 Lane Follower")
    parser.add_argument("--preview", action="store_true",
                        help="Camera + detection only — no motor commands")
    parser.add_argument("--test-image", type=str, default=None,
                        help="Run pipeline on a static image file")
    return parser.parse_args()


# ════════════════════════════════════════════════════════════════════
#  STATIC IMAGE TEST
# ════════════════════════════════════════════════════════════════════

def test_on_image(path):
    """Run the detection pipeline on a single image and display result."""
    img = cv2.imread(path)
    if img is None:
        print(f"[Error] Cannot read image: {path}")
        return

    detector   = LaneDetector()
    controller = LaneController()
    vis        = Visualizer()

    overlay = detector.process(img, depth=None)
    controller.compute(detector)

    frame = vis.render(overlay, detector, controller, odom=None, fps=0)

    # save result
    out_path = path.replace(".", "_detected.")
    cv2.imwrite(out_path, frame)
    print(f"[Test] Result saved → {out_path}")

    cv2.namedWindow("Lane Detection Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Lane Detection Test", 1280, 720)
    cv2.imshow("Lane Detection Test", frame)
    print("[Test] Press any key to exit …")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── static test mode ────────────────────────────────────────────
    if args.test_image:
        test_on_image(args.test_image)
        return

    preview_only = args.preview or not HAS_QCAR

    # ── init hardware ───────────────────────────────────────────────
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   QCar 2 — Industry-Level Lane Detection & Following   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    camera     = SafeCamera3D(verbose=True)
    detector   = LaneDetector()
    controller = LaneController()
    vis        = Visualizer()

    car   = None
    gpad  = None
    odom  = None

    if not preview_only:
        car  = QCar()
        odom = EncoderOdometry()
        odom.reset(car)
        print("[QCar] Initialised")

        if HAS_GAMEPAD:
            try:
                gpad = gamepadViaTarget(1)
                gpad.read()
                print("[Gamepad] Connected")
            except Exception as e:
                print(f"[Gamepad] Not available: {e}")
                gpad = None
    else:
        print("[Mode] Preview only — no motor commands will be sent.")

    # ── state ───────────────────────────────────────────────────────
    manual_mode = False
    paused      = False
    frame_count = 0
    fps         = 0
    fps_timer   = time.time()

    print()
    print("Controls: ESC=quit  G=manual  P=pause  C=cam-reset  S=screenshot")
    print("───────────────────────────────────────────────────────────")
    print()

    try:
        while True:
            loop_t0 = time.time()

            # ── read camera ─────────────────────────────────────────
            rgb, depth = camera.read()

            if rgb is None:
                if car:
                    neutral(car)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                continue

            # ── FPS counter ─────────────────────────────────────────
            frame_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps = frame_count
                frame_count = 0
                fps_timer = now

            # ── pause ───────────────────────────────────────────────
            if paused:
                frame = rgb.copy()
                cv2.putText(frame, "PAUSED", (rgb.shape[1]//2 - 80, rgb.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                key = vis.show(frame)
                if key == ord('p') or key == ord('P'):
                    paused = False
                elif key == 27:
                    break
                if car:
                    neutral(car)
                continue

            # ── lane detection ──────────────────────────────────────
            overlay = detector.process(rgb, depth)

            # ── encoder odometry ────────────────────────────────────
            odom_data = None
            if odom and car:
                odom_data = odom.update(car)

            # ── control ─────────────────────────────────────────────
            if manual_mode and gpad:
                # gamepad manual override
                gpad.read()
                speed = 0.066 * gpad.RT
                steer = 0.5 * gpad.LLA
                mtr_cmd = np.array([speed, steer], dtype=np.float64)
                leds = LEDS_DEFAULT.copy()
                controller.speed_cmd = speed
                controller.steer_cmd = steer
                controller.steer_angle_deg = np.degrees(steer)
                controller.emergency_stop = False
            else:
                mtr_cmd, leds = controller.compute(detector)

            # ── actuate ─────────────────────────────────────────────
            if car and not preview_only:
                try:
                    car.read_write_std(mtr_cmd, leds)
                except Exception:
                    neutral(car)

            # ── visualise ───────────────────────────────────────────
            display = vis.render(overlay, detector, controller, odom_data, fps, manual_mode)
            key = vis.show(display)

            # ── keyboard handling ───────────────────────────────────
            if key == 27:                              # ESC
                break
            elif key == ord('g') or key == ord('G'):   # toggle manual
                manual_mode = not manual_mode
                print(f"[Mode] {'MANUAL' if manual_mode else 'AUTO'}")
                if not manual_mode:
                    controller.pid.reset()
            elif key == ord('p') or key == ord('P'):   # pause
                paused = True
                print("[Paused]")
            elif key == ord('c') or key == ord('C'):   # camera reset
                camera.force_reset()
            elif key == ord('s') or key == ord('S'):   # screenshot
                save_screenshot(display)

            # ── loop timing ─────────────────────────────────────────
            elapsed = time.time() - loop_t0
            target_dt = 1.0 / CAMERA_FPS
            sleep_s = target_dt - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("\n[Interrupted]")

    finally:
        print("\n[Shutdown] Cleaning up …")
        if car:
            neutral(car)
            try:
                car.terminate()
            except Exception:
                pass
        if gpad:
            try:
                gpad.terminate()
            except Exception:
                pass
        camera.terminate()
        vis.destroy()
        print("[Done]")


if __name__ == "__main__":
    main()
