"""Shared data structures passed between SafePath pipeline stages."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class Position(str, Enum):
    LEFT = "LEFT"
    CENTER = "CENTER"
    RIGHT = "RIGHT"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class Detection:
    """Raw output of the object detector for a single bounding box."""
    class_name: str                          # e.g. "person"
    category: str                            # internal risk category, e.g. "person"/"vehicle"
    confidence: float
    bbox: Tuple[int, int, int, int]          # x1, y1, x2, y2 in pixel coords
    class_id: Optional[int] = None           # COCO class index (0-79)
    track_id: Optional[int] = None           # Persistent tracking ID across frames
    velocity: Optional[Tuple[float, float]] = None  # (vx, vy) in px/s
    is_approaching: Optional[bool] = None    # True if closing distance towards user
    time_to_collision: Optional[float] = None  # Estimated seconds to collision
    timestamp: Optional[float] = None        # Frame capture / detection timestamp


@dataclass
class DepthEstimate:
    """
    Distance is ALWAYS approximate. `source` tells the caller which
    estimator produced it so the UI/voice layer can be honest about
    confidence if needed.
    """
    distance_m: Optional[float]   # None if it could not be estimated
    source: str                   # "geometric" | "midas" | "unknown"


@dataclass
class ScoredDetection:
    """A Detection enriched with depth, position, and a risk score."""
    detection: Detection
    distance_m: Optional[float]
    depth_source: str
    position: Position
    risk_score: float             # 0-100
    risk_level: RiskLevel


@dataclass
class DetectionResult:
    """Batch or single-frame detection response with pipeline telemetry."""
    detections: list[Detection]
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_ms: float = 0.0
    device: str = "cpu"
    frame_shape: Tuple[int, int] = (480, 640)
