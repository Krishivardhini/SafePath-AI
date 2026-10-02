"""
SafePath configuration.

Everything tunable lives here so the rest of the codebase never hardcodes
thresholds. Change values here rather than inside core/ modules.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

# Pretrained, lightweight. Ultralytics auto-downloads this on first run.
# Swap for a fine-tuned .pt or .onnx later without touching any other module.
YOLO_MODEL_PATH = "yolov8n.pt"

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
DETECTOR_DEVICE = "auto"         # "auto" | "cuda" | "mps" | "cpu"
INFERENCE_IMG_SIZE = 480        # Optimized YOLO input resolution for real-time edge/cloud inference
HALF_PRECISION = True           # Use FP16 on CUDA/MPS devices if supported
WARMUP_ON_LOAD = True           # Pre-warm model with dummy tensor to eliminate cold-start lag

# Known classes mapped to risk categories. The detector still exposes all
# model classes; any class not listed here receives category "other".
TARGET_CLASSES: Dict[str, str] = {
    # Pedestrians & dynamic agents
    "person": "person",
    
    # Vehicles & transit hazards
    "bicycle": "vehicle",
    "car": "vehicle",
    "motorcycle": "vehicle",
    "airplane": "vehicle",
    "bus": "vehicle",
    "train": "vehicle",
    "truck": "vehicle",
    "boat": "vehicle",
    
    # Structural & street furniture
    "traffic light": "structural",
    "fire hydrant": "structural",
    "stop sign": "structural",
    "parking meter": "structural",
    "bench": "furniture",
    
    # Animals
    "dog": "animal",
    "cat": "animal",
    "horse": "animal",
    "sheep": "animal",
    "cow": "animal",
    
    # Obstacles & personal items
    "backpack": "small_object",
    "umbrella": "small_object",
    "handbag": "small_object",
    "suitcase": "small_object",
    "bottle": "small_object",
    "cup": "small_object",
    "skateboard": "small_object",
    
    # Indoor furniture & barriers
    "chair": "furniture",
    "couch": "furniture",
    "potted plant": "furniture",
    "bed": "furniture",
    "dining table": "furniture",
    "toilet": "furniture",
}


# ---------------------------------------------------------------------------
# Camera / frame
# ---------------------------------------------------------------------------

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


# ---------------------------------------------------------------------------
# Depth estimation
# ---------------------------------------------------------------------------

# "geometric" (default, lightweight, no extra download) or "midas"
# (heavier monocular depth model, falls back to geometric if unavailable).
DEPTH_BACKEND = "geometric"

# Rough focal length in PIXELS for the geometric (pinhole) estimator:
#   distance_m = (real_world_width_m * focal_length_px) / pixel_width_px
# This is a per-camera constant. The default below is a reasonable guess
# for a typical laptop/phone webcam at 640px width but WILL be inaccurate
# until you calibrate it -- see README "Calibration" section.
FOCAL_LENGTH_PX = 600.0

# Average real-world width (meters) used by the geometric estimator.
# These are rough population averages, not measurements of the specific
# object in frame -- another reason distance is always "approximate".
KNOWN_OBJECT_WIDTH_M: Dict[str, float] = {
    "person": 0.45,
    "bicycle": 0.60,
    "car": 1.80,
    "motorcycle": 0.80,
    "bus": 2.55,
    "truck": 2.50,
    "airplane": 15.0,
    "train": 3.00,
    "boat": 2.40,
    "traffic light": 0.35,
    "fire hydrant": 0.35,
    "stop sign": 0.75,
    "parking meter": 0.25,
    "chair": 0.45,
    "couch": 1.80,
    "bed": 1.50,
    "dining table": 1.10,
    "toilet": 0.50,
    "backpack": 0.30,
    "umbrella": 0.90,
    "handbag": 0.30,
    "suitcase": 0.45,
    "bottle": 0.08,
    "cup": 0.08,
    "skateboard": 0.20,
    "bench": 1.20,
    "potted plant": 0.35,
    "dog": 0.35,
    "cat": 0.25,
    "horse": 0.80,
    "sheep": 0.50,
    "cow": 0.90,
}

# Only used by the MiDaS backend: name of the torch.hub model variant.
MIDAS_MODEL_TYPE = "MiDaS_small"


# ---------------------------------------------------------------------------
# Position classification
# ---------------------------------------------------------------------------

# Fraction of frame width considered the "center" zone (rest split L/R).
CENTER_ZONE_FRACTION = 0.34


# ---------------------------------------------------------------------------
# Risk scoring
# ---------------------------------------------------------------------------

# Base weight per risk category (0-1). Combined with distance, walking-path
# position, and detection confidence. A detected person is not inherently an
# urgent obstacle; proximity and whether they are in the user's path decide
# that.
CATEGORY_WEIGHT: Dict[str, float] = {
    "person": 0.75,
    "vehicle": 0.85,
    "animal": 0.7,
    "furniture": 0.5,
    "small_object": 0.35,
    "structural": 1.0,  # reserved for future stairs/pole detections
}

POSITION_WEIGHT: Dict[str, float] = {
    "CENTER": 1.0,
    # Objects outside the centre walking path should not dominate the alert.
    "LEFT": 0.25,
    "RIGHT": 0.25,
}

# Distance (meters) beyond which an obstacle contributes ~0 urgency.
MAX_RELEVANT_DISTANCE_M = 6.0

# Risk score (0-100) thresholds -> RiskLevel.
RISK_LEVEL_THRESHOLDS = {
    "CRITICAL": 75,
    "HIGH": 50,
    "MEDIUM": 25,
    # below MEDIUM => LOW
}


# ---------------------------------------------------------------------------
# Alerting (TTS + haptic) behaviour
# ---------------------------------------------------------------------------

# Minimum seconds between two spoken alerts for the SAME risk level, to
# avoid nagging. A jump to CRITICAL always interrupts immediately.
ALERT_COOLDOWN_SECONDS = 2.0

TTS_RATE_WPM = 175
TTS_VOLUME = 1.0


@dataclass
class RuntimeFlags:
    """Flags typically set from CLI args in main.py."""
    source: str = "0"          # "0" = default webcam, or path to a video file
    depth_backend: str = DEPTH_BACKEND
    enable_audio: bool = True
    enable_display: bool = True
    enable_haptic: bool = True
