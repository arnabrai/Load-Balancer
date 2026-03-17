"""
config.py — Centralised tunables for the QCar 2 Lane-Detection & Following System.

Every magic number lives here. Grouped by subsystem so they are easy to find
and adjust during on-track calibration.
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════════════
#  CAMERA
# ═══════════════════════════════════════════════════════════════════════
CAMERA_MODE        = "RGB&DEPTH"
CAMERA_WIDTH       = 1280
CAMERA_HEIGHT      = 720
CAMERA_FPS         = 30.0
CAMERA_DEVICE_ID   = "0"

# SafeCamera3D recovery
CAMERA_FAIL_RESET_THRESHOLD = 8      # consecutive bad frames before reset
CAMERA_MAX_NO_GOOD_SECS     = 2.5    # seconds without a good frame → reset

# ═══════════════════════════════════════════════════════════════════════
#  DEPTH FILTERING
# ═══════════════════════════════════════════════════════════════════════
DEPTH_MIN_M = 0.15    # ignore anything closer (camera noise zone)
DEPTH_MAX_M = 1.50    # ignore anything farther (not road surface)

# ═══════════════════════════════════════════════════════════════════════
#  COLOUR-SPACE THRESHOLDS  (HSV / LAB)
# ═══════════════════════════════════════════════════════════════════════
# Yellow line — two HSV ranges for robustness under varying lighting
YELLOW_HSV_LOW_1  = np.array([15,  60,  80])
YELLOW_HSV_HIGH_1 = np.array([45, 255, 255])
YELLOW_HSV_LOW_2  = np.array([15,  40,  60])
YELLOW_HSV_HIGH_2 = np.array([45, 255, 200])

# White line
WHITE_HSV_LOW     = np.array([0,    0, 190])
WHITE_HSV_HIGH    = np.array([180, 60, 255])

# White-glare rejection
GLARE_HSV_LOW     = np.array([0,   0, 220])
GLARE_HSV_HIGH    = np.array([180, 60, 255])

# LAB B-channel (captures yellow regardless of brightness)
LAB_USE_OTSU      = True

# Morphology kernel size
MORPH_KERNEL_SIZE  = 5

# ═══════════════════════════════════════════════════════════════════════
#  REGION OF INTEREST (trapezoidal, normalised 0-1)
# ═══════════════════════════════════════════════════════════════════════
# Vertices as fractions of (width, height): TL, TR, BR, BL
ROI_VERTICES_NORM = np.array([
    [0.10, 0.55],   # top-left
    [0.90, 0.55],   # top-right
    [1.00, 1.00],   # bottom-right
    [0.00, 1.00],   # bottom-left
], dtype=np.float32)

# ═══════════════════════════════════════════════════════════════════════
#  BIRD'S-EYE VIEW TRANSFORM
# ═══════════════════════════════════════════════════════════════════════
# Source quadrilateral (normalised fractions of width, height)
BEV_SRC_NORM = np.array([
    [0.38, 0.60],   # top-left
    [0.62, 0.60],   # top-right
    [0.90, 0.95],   # bottom-right
    [0.10, 0.95],   # bottom-left
], dtype=np.float32)

# Destination quadrilateral (normalised)
BEV_DST_NORM = np.array([
    [0.20, 0.00],
    [0.80, 0.00],
    [0.80, 1.00],
    [0.20, 1.00],
], dtype=np.float32)

# BEV output resolution
BEV_WIDTH  = 640
BEV_HEIGHT = 480

# ═══════════════════════════════════════════════════════════════════════
#  SLIDING WINDOW SEARCH
# ═══════════════════════════════════════════════════════════════════════
SW_NUM_WINDOWS     = 9       # number of horizontal bands
SW_MARGIN_PX       = 80      # half-width of each search window
SW_MIN_PIX         = 40      # minimum white pixels to re-centre window
SW_HIST_SMOOTH_K   = 15      # Gaussian smoothing kernel for histogram

# ═══════════════════════════════════════════════════════════════════════
#  POLYNOMIAL FITTING & SMOOTHING
# ═══════════════════════════════════════════════════════════════════════
POLY_ORDER         = 2       # 2nd-degree polynomial  x = a*y² + b*y + c
EMA_ALPHA          = 0.30    # 0 = only history, 1 = only current frame
MIN_LANE_PIXELS    = 300     # minimum pixels to consider a lane valid

# Curvature conversion (pixels → metres — approximate)
YM_PER_PIX         = 0.025   # metres per pixel in y direction (BEV)
XM_PER_PIX         = 0.008   # metres per pixel in x direction (BEV)

# ═══════════════════════════════════════════════════════════════════════
#  PID STEERING CONTROLLER
# ═══════════════════════════════════════════════════════════════════════
PID_KP             = 0.008   # proportional gain
PID_KI             = 0.0003  # integral gain
PID_KD             = 0.004   # derivative gain
PID_INTEGRAL_MAX   = 50.0    # anti-windup clamp
STEER_MAX_RAD      = 0.50    # QCar hardware limit ≈ ±28°
STEER_DEADBAND     = 2.0     # px offset below which steer = 0

# ═══════════════════════════════════════════════════════════════════════
#  SPEED CONTROLLER
# ═══════════════════════════════════════════════════════════════════════
SPEED_BASE         = 0.080   # default forward speed (PWM fraction)
SPEED_MIN          = 0.055   # minimum speed (don't stall)
SPEED_MAX          = 0.095   # maximum speed
SPEED_CURVE_GAIN   = 0.05    # speed reduction ∝ 1/|radius|
SPEED_TURN_COS_MIN = 0.50   # floor for cos-based turn-speed reduction

# ═══════════════════════════════════════════════════════════════════════
#  ENCODER / ODOMETRY
# ═══════════════════════════════════════════════════════════════════════
ENCODER_TICKS_PER_REV = 31_844.0
WHEEL_DIAMETER_M      = 0.066
WHEEL_CIRCUMFERENCE_M = np.pi * WHEEL_DIAMETER_M
ODOM_EMA_ALPHA        = 0.35   # velocity smoothing
ODOM_DT_MIN           = 0.004  # reject impossibly short dt
ODOM_DT_MAX           = 0.350  # reject stale dt
ODOM_V_HARD_MAX       = 6.0    # reject unrealistic speed

# ═══════════════════════════════════════════════════════════════════════
#  SAFETY
# ═══════════════════════════════════════════════════════════════════════
NO_LANE_STOP_FRAMES = 15     # frames with no lane before emergency stop
THROTTLE_SATURATION = 0.20   # absolute max motor command

# LEDs: [Lf_left, Lf_right, Lr_left, Lr_right, user0..3]
LEDS_DEFAULT = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.float64)

# ═══════════════════════════════════════════════════════════════════════
#  HUD / DISPLAY
# ═══════════════════════════════════════════════════════════════════════
HUD_FONT            = 0       # cv2.FONT_HERSHEY_SIMPLEX
HUD_FONT_SCALE      = 0.55
HUD_THICKNESS       = 2
HUD_LINE_HEIGHT     = 28
HUD_X               = 15
HUD_Y               = 30
HUD_BG_ALPHA        = 0.45    # semi-transparent background

OVERLAY_LANE_COLOR  = (0, 200, 80)     # green fill between lanes
OVERLAY_LEFT_COLOR  = (0, 255, 255)    # yellow left boundary
OVERLAY_RIGHT_COLOR = (255, 255, 255)  # white right boundary
OVERLAY_CENTER_COLOR= (255, 200, 0)    # cyan centre line
OVERLAY_THICKNESS   = 3

MINIMAP_SIZE        = (200, 160)       # BEV inset (w, h)
MINIMAP_POSITION    = "bottom-right"

WINDOW_NAME         = "QCar 2 — Lane Follower"
WINDOW_SIZE         = (1280, 720)
