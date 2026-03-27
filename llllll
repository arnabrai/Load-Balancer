#!/usr/bin/env python3
"""
QCar2 Robust Reactive Obstacle Avoidance (Production Grade)
Paradigm: Potential Field with Stability Layer & TTC Awareness
"""

import os
import sys
import numpy as np
import time
import math

# Hardware Compatibility Layer
sys.path.insert(0, '/home/nvidia/Documents/Quanser/0_libraries/python')
from pal.products.qcar import QCar, QCarLidar, IS_PHYSICAL_QCAR

class Config:
    # --- Control Frequencies ---
    LOOP_HZ                 = 30
    DT                      = 1.0 / LOOP_HZ

    # --- Distance Zones (Meters) ---
    D_SAFE                  = 1.5
    D_CAUTION               = 1.0
    D_CRITICAL              = 0.6
    D_STOP                  = 0.4
    D_RESUME                = 0.5  # Hysteresis

    # --- Speed Parameters ---
    THROTTLE_CRUISE         = 0.12
    THROTTLE_MIN            = 0.05  # Speed floor for Ackermann authority
    TTC_THRESHOLD           = 1.5   # Seconds until collision

    # --- Steering Parameters ---
    MAX_STEER               = 0.5   # Radians
    STEER_P_GAIN            = 1.5
    STEER_DEADZONE          = 0.02
    STEER_SMOOTH_ALPHA      = 0.3   # Low-pass filter (0.3 = 30% new, 70% old)
    THROTTLE_SMOOTH_ALPHA   = 0.2

    # --- LiDAR Ranges ---
    FRONT_ARC               = 45.0  # +/- degrees
    MIN_LIDAR_DIST          = 0.05
    MAX_LIDAR_DIST          = 5.0
    MIN_VALID_POINTS        = 5     # Sparsity check
    LIDAR_ANGLE_OFFSET      = 180.0 # Rotation in DEGREES (QCar2 typical: 180)
    LIDAR_REVERSE           = True  # Flip if left/right are swapped

class ReactiveController:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.direction_persistence = 0.0 # -1 for left, 1 for right
        self.is_emergency_stop = False

    def process_lidar(self, angles, distances):
        """Filters LiDAR points within the front arc and returns valid data."""
        if distances is None or len(distances) == 0:
            return None, None

        # Convert to numpy for faster processing
        raw_angles = np.array(angles)
        raw_distances = np.array(distances)

        # 1. Flip angles if reversed
        if self.cfg.LIDAR_REVERSE:
            raw_angles = (2 * math.pi - raw_angles) % (2 * math.pi)
        
        # 2. Add offset (e.g. 180 deg) and normalize to [-pi, pi]
        # 0 should be FRONT
        adj_angles = (raw_angles + math.radians(self.cfg.LIDAR_ANGLE_OFFSET))
        adj_angles = (adj_angles + math.pi) % (2 * math.pi) - math.pi
        
        mask = (np.abs(np.degrees(adj_angles)) <= self.cfg.FRONT_ARC) & \
               (raw_distances > self.cfg.MIN_LIDAR_DIST) & \
               (raw_distances < self.cfg.MAX_LIDAR_DIST)

        return adj_angles[mask], raw_distances[mask]

    def compute_control(self, angles, distances):
        """Core Reactive Logic: Potential Field + Stability Layer"""
        if angles is None or len(angles) < self.cfg.MIN_VALID_POINTS:
            # Sparsity Fallback: If no data, be cautious (slow down, center steering)
            target_throttle = self.cfg.THROTTLE_MIN
            target_steer = 0.0
            return target_throttle, target_steer, "SPARSE_DATA"

        min_dist = np.min(distances)
        
        # 1. Emergency Stop Hysteresis
        if min_dist < self.cfg.D_STOP:
            self.is_emergency_stop = True
        elif min_dist > self.cfg.D_RESUME:
            self.is_emergency_stop = False

        if self.is_emergency_stop:
            return 0.0, 0.0, "EMERGENCY_STOP"

        # 2. Repulsion Model
        # Wi = (max(0, D_thresh - di)^2) * cos(theta_i)
        weights = (np.maximum(0, self.cfg.D_SAFE - distances)**2) * np.cos(angles)
        
        # Vi = weight * [-cos(theta), -sin(theta)]
        # Summing vectors (V_repulsion)
        v_repulse_x = np.sum(weights * -np.cos(angles))
        v_repulse_y = np.sum(weights * -np.sin(angles))

        # 3. Lateral Imbalance Correction
        # sum_left vs sum_right weights
        left_mask = angles > 0.02
        right_mask = angles < -0.02
        sum_left = np.sum(weights[left_mask]) if np.any(left_mask) else 0.0
        sum_right = np.sum(weights[right_mask]) if np.any(right_mask) else 0.0
        # If sum_left is high, we want a NEGATIVE bias (Right)
        imbalance_bias = 0.3 * (sum_right - sum_left) 
        v_repulse_y += imbalance_bias

        # 4. Forward Bias + Direction Persistence
        v_total_x = 1.0 + v_repulse_x
        v_total_y = v_repulse_y
        
        # Persistence: If dead-ahead, nudge towards previous avoidance direction
        if abs(v_total_y) < 0.1 and abs(self.direction_persistence) > 0.1:
            v_total_y += 0.2 * self.direction_persistence

        # 5. Steering Mapping
        target_delta = math.atan2(v_total_y, v_total_x)
        # Scale steering by speed: Faster = less aggressive (Ackermann protection)
        speed_factor = 1.0 - (self.prev_throttle / self.cfg.THROTTLE_CRUISE) * 0.3
        target_steer = np.clip(self.cfg.STEER_P_GAIN * target_delta * speed_factor, 
                               -self.cfg.MAX_STEER, self.cfg.MAX_STEER)

        # Update persistence direction
        if abs(target_steer) > 0.1:
            self.direction_persistence = np.sign(target_steer)

        # 6. Adaptive Speed (Zones + TTC)
        if min_dist > self.cfg.D_SAFE:
            target_throttle = self.cfg.THROTTLE_CRUISE
        elif min_dist > self.cfg.D_CRITICAL:
            # Linear slow-down between SAFE and CRITICAL
            ratio = (min_dist - self.cfg.D_CRITICAL) / (self.cfg.D_SAFE - self.cfg.D_CRITICAL)
            target_throttle = self.cfg.THROTTLE_MIN + ratio * (self.cfg.THROTTLE_CRUISE - self.cfg.THROTTLE_MIN)
        else:
            # Critical Zone: Speed Floor
            target_throttle = self.cfg.THROTTLE_MIN

        # TTC Check: ttc = d / v (v approx throttle * MaxSpeed)
        # MaxSpeed is roughly 2.0 m/s for QCar2 at 1.0 throttle
        est_v = max(self.prev_throttle * 2.0, 0.05)
        ttc = min_dist / est_v
        if ttc < self.cfg.TTC_THRESHOLD:
            target_throttle = min(target_throttle, self.cfg.THROTTLE_MIN * 1.2)

        # 7. Stability Layer (Smoothing & Deadzone)
        # Deadzone
        if abs(target_steer) < self.cfg.STEER_DEADZONE:
            target_steer = 0.0
            
        # LPF Steering
        self.prev_steer = (self.cfg.STEER_SMOOTH_ALPHA * target_steer) + \
                          (1.0 - self.cfg.STEER_SMOOTH_ALPHA) * self.prev_steer
                          
        # LPF Throttle
        self.prev_throttle = (self.cfg.THROTTLE_SMOOTH_ALPHA * target_throttle) + \
                             (1.0 - self.cfg.THROTTLE_SMOOTH_ALPHA) * self.prev_throttle

        status = f"DIST:{min_dist:.2f}m | STEER:{self.prev_steer:.2f} | THR:{self.prev_throttle:.2f}"
        return self.prev_throttle, self.prev_steer, status

def main():
    cfg = Config()
    ctrl = ReactiveController(cfg)
    
    try:
        # Initialize Hardware
        myLidar = QCarLidar(numMeasurements=720, rangingDistanceMode=2) # Higher res for production
        with QCar(readMode=1, frequency=cfg.LOOP_HZ) as myCar:
            print("🚀 Reactive Obstacle Avoidance Started...")
            t_start = time.time()
            last_telemetry = 0
            
            while time.time() - t_start < 1200: # 20 min run
                # 1. Sense
                myCar.read()
                myLidar.read()
                
                ang, dist = ctrl.process_lidar(myLidar.angles, myLidar.distances)
                
                # 2. Think & Act
                th, st, info = ctrl.compute_control(ang, dist)
                
                # 3. Write Commands
                # Note: Headlights ON for safety, blinkers used during avoidance
                leds = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=float) # Headlights
                if abs(st) > 0.2:
                    if st > 0: leds = np.array([1, 0, 1, 0, 0, 0, 0, 0], dtype=float) # Left
                    else:    leds = np.array([0, 1, 0, 1, 0, 0, 0, 0], dtype=float) # Right
                
                myCar.write(th, st, leds)
                
                # Telemetry
                if time.time() - last_telemetry > 0.3:
                    print(f"[{time.time()-t_start:6.1f}s] {info}")
                    last_telemetry = time.time()
                
                # Sync loop
                time.sleep(max(0, cfg.DT - (time.time() - t_start % cfg.DT)))

    except KeyboardInterrupt:
        print("\n🛑 Stopped by User")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        if 'myLidar' in locals():
            myLidar.terminate()
        print("🏁 Cleanup complete.")

if __name__ == "__main__":
    main()
