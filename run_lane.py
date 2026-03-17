#!/usr/bin/env python3
"""
run_lane.py — QCar2 Autonomous Lane Following
Integrates RealSense camera + LaneDetector + PID + QCar2 hardware.

Usage:
    python3 run_lane.py            # full autonomous mode
    python3 run_lane.py --preview  # detection only, no motors
"""

import sys
import os
import time
import argparse
import threading
import numpy as np
import cv2

# ── add scripts folder to path ──────────────────────────────────────
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

from lane_detector import LaneDetector
from controller   import LaneController
from visualizer   import Visualizer
from config       import LEDS_DEFAULT, CAMERA_FPS

# ── ROS2 + RealSense ────────────────────────────────────────────────
import rclpy
from rclpy.node import Node
from rclpy.qos  import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, LaserScan

# ── QCar2 hardware ───────────────────────────────────────────────────
try:
    import sys as _sys
    _sys.path.insert(0, '/home/nvidia/Documents/Quanser/0_libraries/python')
    from pal.products.qcar import QCar
    HAS_QCAR = True
except ImportError:
    HAS_QCAR = False
    print("[run_lane] WARNING: QCar PAL not found — preview mode only.")


# ════════════════════════════════════════════════════════════════════
#  ROS2 Sensor Node
# ════════════════════════════════════════════════════════════════════

class SensorNode(Node):
    """Subscribes to RealSense RGB, Depth, and LiDAR."""

    def __init__(self):
        super().__init__('run_lane_sensors')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._lock  = threading.Lock()
        self._rgb   = None
        self._depth = None
        self._lidar_dist = 99.0  # meters

        self.create_subscription(Image,     '/camera/color_image', self._rgb_cb,   qos)
        self.create_subscription(Image,     '/camera/depth_image', self._depth_cb, qos)
        self.create_subscription(LaserScan, '/scan',               self._lidar_cb, qos)

    def _rgb_cb(self, msg):
        if len(msg.data) == 0:
            return
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, -1)
        with self._lock:
            self._rgb = frame.copy()

    def _depth_cb(self, msg):
        if len(msg.data) == 0:
            return
        frame = np.frombuffer(msg.data, dtype=np.uint16).reshape(
            msg.height, msg.width).astype(np.float32) * 0.001
        with self._lock:
            self._depth = frame

    def _lidar_cb(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)
        angles = np.linspace(
            np.degrees(msg.angle_min),
            np.degrees(msg.angle_max),
            len(ranges)
        )
        # front 30 degree arc
        mask = (np.abs(angles) < 15) & (ranges > 0.05) & (ranges < 10.0)
        with self._lock:
            self._lidar_dist = float(np.min(ranges[mask])) if mask.any() else 99.0

    def read(self):
        with self._lock:
            rgb   = self._rgb.copy()   if self._rgb   is not None else None
            depth = self._depth.copy() if self._depth is not None else None
            dist  = self._lidar_dist
        return rgb, depth, dist


# ════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--preview',   action='store_true',
                   help='Camera + detection only — no motor commands')
    p.add_argument('--stop',      type=float, default=0.0,
                   help='Stop at this distance in cm (0 = disabled)')
    return p.parse_args()


def neutral(car):
    try:
        car.write(0.0, 0.0, np.zeros(8))
    except Exception:
        pass


def main():
    args        = parse_args()
    preview     = args.preview or not HAS_QCAR
    stop_dist_m = args.stop / 100.0 if args.stop > 0 else 0.0

    print("╔══════════════════════════════════════════════════════╗")
    print("║   QCar2 — Autonomous Lane Following                 ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Mode    : {'PREVIEW (no motors)' if preview else 'AUTONOMOUS'}")
    print(f"  Stop at : {args.stop:.0f} cm" if stop_dist_m > 0 else "  Stop at : disabled")
    print()

    # ── init ROS2 ───────────────────────────────────────────────────
    rclpy.init()
    sensors     = SensorNode()
    ros_thread  = threading.Thread(target=rclpy.spin, args=(sensors,), daemon=True)
    ros_thread.start()
    print("[ROS2] Sensor node started.")

    # ── init perception & control ───────────────────────────────────
    detector   = LaneDetector()
    controller = LaneController()
    vis        = Visualizer()

    # ── init QCar2 hardware ─────────────────────────────────────────
    car = None
    if not preview:
        try:
            car = QCar(readMode=1, frequency=100)
            car.__enter__()
            print("[QCar2] Hardware connected! ✅")
        except Exception as e:
            print(f"[QCar2] Hardware failed: {e} — switching to preview mode.")
            preview = True

    # ── wait for first frame ─────────────────────────────────────────
    print("[Camera] Waiting for RealSense frames...")
    for _ in range(50):
        rgb, _, _ = sensors.read()
        if rgb is not None:
            break
        time.sleep(0.1)
    if rgb is None:
        print("[ERROR] No camera frames received! Is 'ros2 run qcar2_nodes rgbd' running?")
        return

    print("[Camera] RealSense ready! ✅")
    print()
    print("Controls: ESC = quit   P = pause   S = screenshot")
    print("─────────────────────────────────────────────────")

    # ── state ───────────────────────────────────────────────────────
    paused      = False
    stopped     = False
    frame_count = 0
    fps         = 0
    fps_timer   = time.time()

    try:
        while True:
            t0 = time.time()

            # ── read sensors ─────────────────────────────────────────
            rgb, depth, lidar_dist = sensors.read()
            if rgb is None:
                time.sleep(0.01)
                continue

            # ── FPS ──────────────────────────────────────────────────
            frame_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps = frame_count
                frame_count = 0
                fps_timer   = now

            # ── pause ────────────────────────────────────────────────
            if paused:
                frame = rgb.copy()
                cv2.putText(frame, "PAUSED — press P to resume",
                            (50, rgb.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                vis.show(frame)
                if car:
                    neutral(car)
                key = cv2.waitKey(30) & 0xFF
                if key == ord('p') or key == ord('P'):
                    paused = False
                elif key == 27:
                    break
                continue

            # ── lane detection ───────────────────────────────────────
            overlay = detector.process(rgb, depth)

            # ── control ──────────────────────────────────────────────
            mtr_cmd, leds = controller.compute(detector)
            speed   = float(mtr_cmd[0])
            steer   = float(mtr_cmd[1])

            # ── distance stop ─────────────────────────────────────────
            if stop_dist_m > 0 and lidar_dist <= stop_dist_m and not stopped:
                stopped = True
                print(f"\n🛑 STOPPED! Object at {lidar_dist*100:.1f}cm "
                      f"(target: {stop_dist_m*100:.0f}cm)")

            if stopped or controller.emergency_stop:
                speed = 0.0
                steer = 0.0
                leds[4] = 1   # hazard lights

            # ── send to hardware ──────────────────────────────────────
            if car and not preview:
                try:
                    car.write(speed, steer, leds)
                except Exception as e:
                    print(f"[QCar2] Write error: {e}")
                    neutral(car)

            # ── visualise ─────────────────────────────────────────────
            display = vis.render(overlay, detector, controller,
                                 odom=None, fps=fps, manual_mode=False)

            # extra HUD — lidar distance + stop status
            h_d, w_d = display.shape[:2]
            dist_txt  = f"LiDAR: {lidar_dist*100:.0f}cm"
            dist_col  = (0, 0, 255) if (stop_dist_m > 0 and lidar_dist <= stop_dist_m) \
                        else (0, 255, 0)
            cv2.putText(display, dist_txt,
                        (w_d - 220, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, dist_col, 2)

            if stopped:
                cv2.putText(display, f"STOPPED at {lidar_dist*100:.0f}cm",
                            (w_d//2 - 160, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            key = vis.show(display)

            # ── keyboard ──────────────────────────────────────────────
            if key == 27:                               # ESC
                break
            elif key == ord('p') or key == ord('P'):   # pause
                paused = True
            elif key == ord('s') or key == ord('S'):   # screenshot
                ts   = time.strftime("%H%M%S")
                path = f"/home/nvidia/Desktop/SDV_workspace/scripts/screenshot_{ts}.jpg"
                cv2.imwrite(path, display)
                print(f"[Screenshot] Saved → {path}")
            elif key == ord('r') or key == ord('R'):   # resume after stop
                stopped = False
                print("[Run] Resumed!")

            # ── loop rate ─────────────────────────────────────────────
            elapsed  = time.time() - t0
            sleep_t  = (1.0 / CAMERA_FPS) - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[Interrupted]")

    finally:
        print("\n[Shutdown] Stopping car safely...")
        if car:
            neutral(car)
            car.__exit__(None, None, None)
        vis.destroy()
        sensors.destroy_node()
        rclpy.shutdown()
        print("[Done] ✅")


if __name__ == "__main__":
    main()
