#!/usr/bin/env python3
"""
=============================================================================
 QCAR2 AUTONOMOUS NAVIGATION SYSTEM v1.0
 SDCS Road Map — Right-Hand Traffic — Small Map
=============================================================================

 Features:
   - Encoder-based distance driving (NOT time-based)
   - PI speed controller with anti-windup
   - A* path planning on node graph
   - go_to_node(start, end) — single call navigation
   - Proper QCar2 PAL API (write, not read_write_std)
   - Safe hardware shutdown on every exit path
   - LED indicators per maneuver
   - Live telemetry every 0.5s

 Map:
   11 nodes, 15 directed edges
   Right-hand traffic, small map configuration
   Coordinate origin: node 0 area (x=0, y=0.13)

 Usage:
   python3 qcar2_navigation.py

 To change route — edit MISSION at bottom of file:
   MISSION = [0, 4, 6, 0]   ← visits node 0 → 4 → 6 → 0

=============================================================================
"""

import os
import sys
import math
import time
import heapq
import numpy as np
from collections import defaultdict

os.environ.pop("DISPLAY",    None)
os.environ.pop("XAUTHORITY", None)
sys.path.insert(0, '/home/nvidia/Documents/Quanser/0_libraries/python')

from pal.products.qcar import QCar, QCarLidar, IS_PHYSICAL_QCAR

if not IS_PHYSICAL_QCAR:
    try:
        import qlabs_setup
        qlabs_setup.setup()
    except ImportError:
        print("[WARN] qlabs_setup not found")


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    # ── Speed controller ──────────────────────────────────────────────────────
    CRUISE_SPEED        = 0.12   # m/s — normal driving
    CURVE_SPEED         = 0.09   # m/s — during turns
    APPROACH_SPEED      = 0.08   # m/s — final approach to node
    MAX_THROTTLE        = 0.25   # hard clamp

    # ── PI speed controller gains ─────────────────────────────────────────────
    KP_SPEED            = 0.15
    KI_SPEED            = 0.08
    MAX_INTEGRAL        = 0.20

    # ── Distance tolerance ────────────────────────────────────────────────────
    DIST_TOLERANCE      = 0.015  # m — stop within 1.5cm of target

    # ── Loop rate ─────────────────────────────────────────────────────────────
    CONTROL_HZ          = 100    # Hz
    LOOP_PERIOD         = 1.0 / 100

    # ── Segment pause between edges ───────────────────────────────────────────
    PAUSE_BETWEEN       = 0.25   # s — brief stop between edge segments

    # ── Battery ───────────────────────────────────────────────────────────────
    BATTERY_WARN        = 10.5
    BATTERY_CRITICAL    = 9.5

    # ── LEDs ──────────────────────────────────────────────────────────────────
    LED_OFF      = np.zeros(8,        dtype=np.float64)
    LED_FWD      = np.array([0,0,0,0,0,0,1,1], dtype=np.float64)  # headlights
    LED_LEFT     = np.array([1,0,1,0,0,0,1,1], dtype=np.float64)  # left indicators
    LED_RIGHT    = np.array([0,1,0,1,0,0,1,1], dtype=np.float64)  # right indicators
    LED_BRAKE    = np.array([0,0,0,0,1,1,1,1], dtype=np.float64)  # brake lights
    LED_HAZARD   = np.array([1,1,1,1,1,1,1,1], dtype=np.float64)  # all on


# ══════════════════════════════════════════════════════════════════════════════
#  ROAD MAP — 11 nodes, 15 directed edges
#  All coordinates in meters. Origin near node 0.
#  Heading in radians: 0=RIGHT, π/2=UP, π=LEFT, -π/2=DOWN
# ══════════════════════════════════════════════════════════════════════════════

PI      = math.pi
HALF_PI = PI / 2

SCALE   = 0.002035
X_OFF   = 1134
Y_OFF   = 2363

def _px(raw):
    """Convert [pixel_x, pixel_y, heading_rad] → (x_m, y_m, heading_rad)"""
    return (
        SCALE * (raw[0] - X_OFF),
        SCALE * (Y_OFF  - raw[1]),
        raw[2]
    )

# Node definitions: [pixel_x, pixel_y, heading_rad]
_RAW_NODES = [
    [1134, 2299, -HALF_PI],          # 0
    [1266, 2323,  HALF_PI],          # 1
    [1688, 2896,  0      ],          # 2
    [1688, 2763,  PI     ],          # 3
    [2242, 2323,  HALF_PI],          # 4
    [2109, 2323, -HALF_PI],          # 5
    [1632, 1822,  PI     ],          # 6
    [1741, 1955,  0      ],          # 7
    [ 766, 1822,  PI     ],          # 8
    [ 766, 1955,  0      ],          # 9
    [ 504, 2589, -42*PI/180],        # 10  diagonal
]

NODES = [_px(r) for r in _RAW_NODES]   # list of (x, y, theta) in meters

# Turning radii
R_OUTER  = 438   * SCALE   # 0.8913 m
R_INNER  = 305.5 * SCALE   # 0.6217 m
R_ONEWAY = 350   * SCALE   # 0.7123 m
WB       = 0.256            # QCar2 wheelbase (meters)

def _steer(radius, left_turn=True):
    """Ackermann steering angle for given radius."""
    angle = math.atan(WB / radius)
    return angle if left_turn else -angle

# ── Edge table ────────────────────────────────────────────────────────────────
# Each edge: (from_node, to_node, segments, label)
# segments: list of (distance_m, steering_rad)
#   — decomposed as: straight approach → arc → straight exit

def _make_edge(n1, n2, radius, name):
    """
    Build 3-segment edge: [straight_in, arc, straight_out]
    using node positions and headings.
    """
    x1, y1, th1 = NODES[n1]
    x2, y2, th2 = NODES[n2]

    # Straight segments
    if radius == 0:
        dist = math.hypot(x2-x1, y2-y1)
        return (n1, n2, [(dist, 0.0)], name, dist)

    # Diagonal nodes (node 10) — approximate as single arc
    if n1 == 10 or n2 == 10:
        dist = math.hypot(x2-x1, y2-y1)
        th_diff = _wrap(th2 - th1)
        steer = _steer(radius, left_turn=(th_diff > 0))
        return (n1, n2, [(dist, steer)], name, dist)

    # Standard 90° corner
    th_diff = _wrap(th2 - th1)
    left    = (th_diff > 0)
    steer   = _steer(radius, left_turn=left)
    arc_len = (PI / 2) * radius

    # Straight lengths before/after arc
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    if abs(math.cos(th1)) > 0.5:   # primarily horizontal heading
        d1 = max(0.0, dx - radius)
        d2 = max(0.0, dy - radius)
    else:                           # primarily vertical heading
        d1 = max(0.0, dy - radius)
        d2 = max(0.0, dx - radius)

    total = d1 + arc_len + d2
    segs  = []
    if d1 > 0.005: segs.append((d1, 0.0))
    segs.append((arc_len, steer))
    if d2 > 0.005: segs.append((d2, 0.0))

    return (n1, n2, segs, name, total)

def _wrap(a):
    """Wrap angle to [-π, π]"""
    while a >  PI: a -= 2*PI
    while a < -PI: a += 2*PI
    return a

# Build all edges
_EDGE_DEFS = [
    (0,  2, R_OUTER,  "N0→N2"),
    (1,  7, R_INNER,  "N1→N7"),
    (1,  8, R_OUTER,  "N1→N8"),
    (2,  4, R_OUTER,  "N2→N4"),
    (3,  1, R_INNER,  "N3→N1"),
    (4,  6, R_OUTER,  "N4→N6"),
    (5,  3, R_INNER,  "N5→N3"),
    (6,  0, R_OUTER,  "N6→N0"),
    (6,  8, 0,        "N6→N8"),
    (7,  5, R_INNER,  "N7→N5"),
    (8, 10, R_ONEWAY, "N8→N10"),
    (9,  0, R_INNER,  "N9→N0"),
    (9,  7, 0,        "N9→N7"),
    (10, 1, R_INNER,  "N10→N1"),
    (10, 2, R_INNER,  "N10→N2"),
]

EDGES = [_make_edge(*e) for e in _EDGE_DEFS]

# Adjacency list for path planning
# adj[from_node] = list of (to_node, edge_index, cost)
ADJ = defaultdict(list)
for i, (frm, to, segs, label, cost) in enumerate(EDGES):
    ADJ[frm].append((to, i, cost))


# ══════════════════════════════════════════════════════════════════════════════
#  A* PATH PLANNER
# ══════════════════════════════════════════════════════════════════════════════

def euclidean_h(node, goal):
    """Heuristic: straight-line distance to goal node."""
    x1, y1, _ = NODES[node]
    x2, y2, _ = NODES[goal]
    return math.hypot(x2-x1, y2-y1)

def astar(start, goal):
    """
    A* search on directed node graph.
    Returns list of edge indices to traverse, or None if no path.
    """
    if start == goal:
        return []

    # Priority queue: (f_cost, g_cost, node, edge_path)
    open_set = [(euclidean_h(start, goal), 0.0, start, [])]
    visited  = {}

    while open_set:
        f, g, node, path = heapq.heappop(open_set)

        if node in visited and visited[node] <= g:
            continue
        visited[node] = g

        if node == goal:
            return path

        for neighbor, edge_idx, cost in ADJ[node]:
            new_g = g + cost
            new_f = new_g + euclidean_h(neighbor, goal)
            if neighbor not in visited or visited[neighbor] > new_g:
                heapq.heappush(open_set, (new_f, new_g, neighbor, path + [edge_idx]))

    return None   # no path found


def plan_mission(node_sequence):
    """
    Plan a full multi-stop mission.
    node_sequence = [A, B, C, D] means go A→B, then B→C, then C→D.
    Returns flat list of edge indices.
    """
    full_path = []
    for i in range(len(node_sequence) - 1):
        src = node_sequence[i]
        dst = node_sequence[i+1]
        segment = astar(src, dst)
        if segment is None:
            print(f"[Planner] ❌ No path from node {src} to node {dst}!")
            return None
        full_path.extend(segment)
        # Print planned route
        edge_names = [EDGES[e][3] for e in segment]
        dist_total = sum(EDGES[e][4] for e in segment)
        print(f"[Planner] {src}→{dst}: {' → '.join(edge_names)}  ({dist_total:.2f}m)")
    return full_path


# ══════════════════════════════════════════════════════════════════════════════
#  PI SPEED CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════

class PISpeedController:
    def __init__(self, kp=Config.KP_SPEED, ki=Config.KI_SPEED):
        self.kp  = kp
        self.ki  = ki
        self._ei = 0.0

    def reset(self):
        self._ei = 0.0

    def update(self, v_measured, v_ref, dt):
        """
        Returns throttle command.
        v_measured: motorTach reading (m/s)
        v_ref:      desired speed (m/s)
        dt:         time step (s)
        """
        error    = v_ref - v_measured
        self._ei = np.clip(self._ei + error * dt,
                           -Config.MAX_INTEGRAL, Config.MAX_INTEGRAL)
        u = self.kp * error + self.ki * self._ei
        return float(np.clip(u, 0.0, Config.MAX_THROTTLE))


# ══════════════════════════════════════════════════════════════════════════════
#  DRIVE PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

class DriveEngine:
    """
    Encoder-based distance driving with PI speed control.
    Uses motorTach integration — NO time.sleep() for distance.
    """

    def __init__(self, qcar: QCar, cfg: Config):
        self.qcar = qcar
        self.cfg  = cfg
        self.pi   = PISpeedController()
        self._tel_time = time.time()

    def _leds_for_steer(self, steer):
        if   steer >  0.05: return Config.LED_LEFT.copy()
        elif steer < -0.05: return Config.LED_RIGHT.copy()
        else:               return Config.LED_FWD.copy()

    def drive_segment(self, dist_m: float, steer_rad: float,
                      label: str = "", speed: float = None):
        """
        Drive exactly dist_m meters with steer_rad steering.
        Uses encoder feedback to measure actual distance.

        Args:
            dist_m    : target distance in meters
            steer_rad : steering command (radians, + = left, - = right)
            label     : display label for telemetry
            speed     : target speed override (None = auto based on steer)
        """
        if dist_m < 0.001:
            return   # skip negligible segments

        # Choose target speed
        if speed is not None:
            v_ref = speed
        elif abs(steer_rad) > 0.05:
            v_ref = Config.CURVE_SPEED
        else:
            v_ref = Config.CRUISE_SPEED

        leds    = self._leds_for_steer(steer_rad)
        dist_done = 0.0
        prev_t    = time.time()
        self.pi.reset()

        print(f"    → {label:20s}  {dist_m:.3f}m  steer={steer_rad:+.3f}  v_ref={v_ref:.2f}")

        while dist_done < dist_m - Config.DIST_TOLERANCE:
            # Timing
            now = time.time()
            dt  = now - prev_t
            prev_t = now
            if dt <= 0: dt = Config.LOOP_PERIOD

            # Read encoder
            self.qcar.read()
            v_meas = abs(self.qcar.motorTach)

            # Integrate distance
            dist_done += v_meas * dt

            # Remaining distance → slow down in last 5cm
            remaining = dist_m - dist_done
            if remaining < 0.05:
                v_ref_actual = max(Config.APPROACH_SPEED,
                                   v_ref * (remaining / 0.05))
            else:
                v_ref_actual = v_ref

            # PI throttle
            throttle = self.pi.update(v_meas, v_ref_actual, dt)

            # Send command
            self.qcar.write(throttle, steer_rad, leds)

            # Telemetry every 0.5s
            if now - self._tel_time >= 0.5:
                print(f"       dist={dist_done:.3f}/{dist_m:.3f}m  "
                      f"v={v_meas:.3f}m/s  thr={throttle:.3f}")
                self._tel_time = now

            # Maintain loop rate
            elapsed = time.time() - now
            slack   = Config.LOOP_PERIOD - elapsed
            if slack > 0:
                time.sleep(slack)

        # Stop after segment
        self.qcar.write(0.0, 0.0, Config.LED_BRAKE.copy())
        time.sleep(Config.PAUSE_BETWEEN)

    def execute_edge(self, edge_idx: int):
        """
        Execute all segments of one map edge.
        """
        frm, to, segments, label, total = EDGES[edge_idx]
        x1,y1,_ = NODES[frm]
        x2,y2,_ = NODES[to]

        print(f"\n  🚗 Edge [{label}]  ({x1:.2f},{y1:.2f}) → ({x2:.2f},{y2:.2f})"
              f"  total={total:.2f}m")

        for i, (dist, steer) in enumerate(segments):
            seg_type = "STRAIGHT" if abs(steer) < 0.01 else \
                       ("LEFT TURN" if steer > 0 else "RIGHT TURN")
            self.drive_segment(dist, steer,
                               label=f"seg{i+1}/{len(segments)} {seg_type}")

        print(f"  ✅ Arrived at node {to}"
              f"  ({x2:.2f}, {y2:.2f})")


# ══════════════════════════════════════════════════════════════════════════════
#  NAVIGATOR — high-level interface
# ══════════════════════════════════════════════════════════════════════════════

class Navigator:
    """
    High-level navigation interface.
    Usage:
        nav = Navigator(qcar)
        nav.go_to_node(0, 6)          # single destination
        nav.run_mission([0, 4, 6, 0]) # multi-stop mission
    """

    def __init__(self, qcar: QCar):
        self.qcar   = qcar
        self.engine = DriveEngine(qcar, Config)
        self.current_node = None

    def go_to_node(self, start: int, goal: int):
        """
        Navigate from start node to goal node using A*.
        """
        print(f"\n{'='*60}")
        print(f"  NAVIGATE: Node {start} → Node {goal}")
        print(f"{'='*60}")

        edge_list = astar(start, goal)

        if edge_list is None:
            print(f"  ❌ No path from {start} to {goal}!")
            return False

        if len(edge_list) == 0:
            print(f"  ✅ Already at node {goal}!")
            self.current_node = goal
            return True

        # Print full plan
        print(f"  Plan: {start}", end="")
        node = start
        for ei in edge_list:
            node = EDGES[ei][1]
            print(f" → {node}", end="")
        print(f"  ({len(edge_list)} edges)")

        total_dist = sum(EDGES[ei][4] for ei in edge_list)
        print(f"  Total distance: {total_dist:.2f}m")

        # Execute
        for i, ei in enumerate(edge_list):
            print(f"\n[{i+1}/{len(edge_list)}]", end="")
            self.engine.execute_edge(ei)

        self.current_node = goal
        print(f"\n{'='*60}")
        print(f"  ✅ MISSION COMPLETE — arrived at node {goal}")
        print(f"{'='*60}")
        return True

    def run_mission(self, node_sequence: list):
        """
        Execute a multi-stop mission.
        node_sequence = [A, B, C, D] → go A→B, then B→C, then C→D.
        """
        print(f"\n{'#'*60}")
        print(f"  MISSION: {' → '.join(str(n) for n in node_sequence)}")
        print(f"{'#'*60}\n")

        # Plan full path upfront
        edge_plan = plan_mission(node_sequence)
        if edge_plan is None:
            return False

        total_dist = sum(EDGES[ei][4] for ei in edge_plan)
        print(f"\n[Mission] Total edges: {len(edge_plan)}")
        print(f"[Mission] Total distance: {total_dist:.2f}m")
        print(f"[Mission] Starting in 3 seconds...")
        time.sleep(3)

        # Execute all edges
        for i, ei in enumerate(edge_plan):
            frm, to, _, label, _ = EDGES[ei]
            print(f"\n[Edge {i+1}/{len(edge_plan)}]  {label}")
            self.engine.execute_edge(ei)
            self._check_battery()

        self.current_node = node_sequence[-1]
        print(f"\n{'#'*60}")
        print(f"  ✅ MISSION COMPLETE")
        print(f"  Final position: Node {self.current_node}")
        print(f"{'#'*60}")
        return True

    def _check_battery(self):
        """Check battery voltage and warn/stop if critical."""
        self.qcar.read()
        v = self.qcar.batteryVoltage
        if v < Config.BATTERY_CRITICAL:
            print(f"\n🔋❌ Battery critical ({v:.1f}V) — STOPPING!")
            raise RuntimeError(f"Battery critical: {v:.1f}V")
        if v < Config.BATTERY_WARN:
            print(f"🔋⚠️  Battery low: {v:.1f}V")


# ══════════════════════════════════════════════════════════════════════════════
#  MAP UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def print_map():
    """Print node and edge summary."""
    print("\n" + "="*60)
    print("  SDCS ROAD MAP — Node Summary")
    print("="*60)
    for i, (x, y, th) in enumerate(NODES):
        deg = math.degrees(th)
        dirs = {-90:"DOWN", 90:"UP", 0:"RIGHT", 180:"LEFT", -42:"DIAG"}
        d = dirs.get(round(deg), f"{deg:.0f}°")
        print(f"  Node {i:2d}: ({x:+.3f}, {y:+.3f})  heading={d}")

    print("\n" + "="*60)
    print("  SDCS ROAD MAP — Edge Summary")
    print("="*60)
    for frm, to, segs, label, total in EDGES:
        print(f"  {label:12s}: {len(segs)} segs, {total:.3f}m total")

    print("\n" + "="*60)
    print("  Common Routes:")
    print("  Outer loop CW  : [0, 2, 4, 6, 0]")
    print("  Inner loop     : [1, 7, 5, 3, 1]")
    print("  Full circuit   : [0, 2, 4, 6, 8, 10, 1, 7, 5, 3, 1, 9, 0]")
    print("="*60 + "\n")


def print_path(start, goal):
    """Print planned path without executing."""
    edges = astar(start, goal)
    if edges is None:
        print(f"No path from {start} to {goal}")
        return
    print(f"\nPath {start} → {goal}:")
    node = start
    total = 0
    for ei in edges:
        frm, to, segs, label, cost = EDGES[ei]
        total += cost
        print(f"  {label}  ({cost:.2f}m)")
        node = to
    print(f"Total: {total:.2f}m, {len(edges)} edges")


# ══════════════════════════════════════════════════════════════════════════════
#  MISSION DEFINITION — Edit this to change what the car does
# ══════════════════════════════════════════════════════════════════════════════

# Common missions:
# Outer loop (clockwise):  [0, 2, 4, 6, 0]
# Inner loop:              [1, 7, 5, 3, 1]
# Cross map:               [0, 4]
# Full circuit:            [9, 7, 5, 3, 1, 8, 10, 2, 4, 6, 0, 9]

MISSION = [0, 2, 4, 6, 0]   # ← CHANGE THIS


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print_map()

    # Show planned path before starting
    print("[Planner] Computing mission path...")
    edge_plan = plan_mission(MISSION)
    if edge_plan is None:
        print("Mission planning failed — check node sequence!")
        return

    total = sum(EDGES[ei][4] for ei in edge_plan)
    print(f"[Planner] ✅ Plan ready: {len(edge_plan)} edges, {total:.2f}m total\n")

    print("Initializing QCar2 hardware...")
    qcar = None

    try:
        qcar = QCar(readMode=1, frequency=Config.CONTROL_HZ)
        qcar.__enter__()
        print("✅ QCar2 connected")

        # Battery check
        qcar.read()
        batt = qcar.batteryVoltage
        print(f"🔋 Battery: {batt:.1f}V")
        if batt < Config.BATTERY_CRITICAL:
            print("❌ Battery too low — aborting!")
            return

        nav = Navigator(qcar)
        nav.run_mission(MISSION)

    except KeyboardInterrupt:
        print("\n\n[Ctrl+C] Stopping...")

    except RuntimeError as e:
        print(f"\n[ERROR] {e}")

    except Exception as e:
        print(f"\n[FATAL] {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n[Shutdown] Stopping hardware safely...")

        # Stop car
        if qcar is not None:
            try:
                qcar.write(0.0, 0.0, Config.LED_BRAKE.copy())
                time.sleep(0.2)
                qcar.write(0.0, 0.0, Config.LED_OFF.copy())
                qcar.__exit__(None, None, None)
                print("  ✅ QCar stopped")
            except Exception:
                pass

        # Stop LiDAR — Quanser hardware stop (same as QCar2_hardware_stop.py)
        try:
            ql = QCarLidar()
            QCar().terminate()
            ql.terminate()
            print("  ✅ LiDAR terminated")
        except Exception as e:
            print(f"  ⚠️  LiDAR stop: {e}")

        print("  ✅ Shutdown complete\n")


if __name__ == "__main__":
    main()
