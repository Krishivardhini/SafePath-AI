"""
Approximate monocular distance estimation.

IMPORTANT: A single camera cannot measure true distance. Everything this
module returns is an ESTIMATE and should always be presented to the user
as approximate (e.g. "about 2 meters", never "2.00 meters").

Two interchangeable backends implement the same interface:

- GeometricDepthEstimator (default): pinhole-camera math using a table of
  average real-world object widths. Cheap, deterministic, no extra model.
- MiDaSDepthEstimator: a real monocular depth network (torch.hub). More
  general (works on any class) but heavier and needs a one-time download.
  If it can't load (e.g. no internet), the factory below automatically
  falls back to the geometric estimator and logs a warning.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from safepath import config
from safepath.core.types import Detection, DepthEstimate
from safepath.utils.logger import get_logger

logger = get_logger(__name__)


class DepthEstimator(ABC):
    @abstractmethod
    def estimate(self, frame: np.ndarray, detection: Detection) -> DepthEstimate:
        ...


class GeometricDepthEstimator(DepthEstimator):
    """
    distance_m = (real_world_width_m * focal_length_px) / pixel_width_px

    Accuracy depends entirely on:
      1. FOCAL_LENGTH_PX being calibrated for the actual camera (see README).
      2. The detected object's real width being close to the population
         average in config.KNOWN_OBJECT_WIDTH_M.
    Treat the output as a rough "near / mid / far" signal, not a ruler.
    """

    def __init__(self, focal_length_px: float = config.FOCAL_LENGTH_PX):
        self._focal_length_px = focal_length_px

    def estimate(self, frame: np.ndarray, detection: Detection) -> DepthEstimate:
        real_width_m = config.KNOWN_OBJECT_WIDTH_M.get(detection.class_name)
        x1, _, x2, _ = detection.bbox
        pixel_width = max(x2 - x1, 1)

        if real_width_m is None:
            return DepthEstimate(distance_m=None, source="unknown")

        distance_m = (real_width_m * self._focal_length_px) / pixel_width
        # Clamp to a sane range; anything absurd means the box/edge case
        # broke the assumption (e.g. heavily occluded object).
        distance_m = max(0.1, min(distance_m, 50.0))
        return DepthEstimate(distance_m=round(distance_m, 1), source="geometric")


class MiDaSDepthEstimator(DepthEstimator):
    """
    Relative monocular depth via MiDaS-small. Produces a per-pixel relative
    depth map (NOT metric distance). We take the median value inside the
    detection's bounding box and pass it through a simple calibration
    factor to produce a rough metric approximation.

    The calibration factor is intentionally crude for the MVP -- this
    backend is offered for future improvement, not as the accurate option.
    """

    def __init__(self, model_type: str = config.MIDAS_MODEL_TYPE):
        self._model_type = model_type
        self._model = None
        self._transform = None
        self._device = None

    def _load(self):
        if self._model is not None:
            return
        import torch  # local import: heavy, optional dependency

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = torch.hub.load("intel-isl/MiDaS", self._model_type)
        self._model.to(self._device).eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self._transform = (
            transforms.small_transform
            if "small" in self._model_type.lower()
            else transforms.default_transform
        )

    def estimate(self, frame: np.ndarray, detection: Detection) -> DepthEstimate:
        import torch

        self._load()

        x1, y1, x2, y2 = detection.bbox
        rgb = frame[:, :, ::-1]  # BGR -> RGB
        input_batch = self._transform(rgb).to(self._device)

        with torch.no_grad():
            prediction = self._model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        h, w = depth_map.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        region = depth_map[y1:y2, x1:x2]

        if region.size == 0:
            return DepthEstimate(distance_m=None, source="unknown")

        relative_depth = float(np.median(region))
        # MiDaS output is inverse-depth-like (larger = closer). This
        # calibration constant is a rough MVP placeholder -- tune per camera.
        calibration_constant = 250.0
        distance_m = calibration_constant / max(relative_depth, 1e-3)
        distance_m = max(0.1, min(distance_m, 50.0))
        return DepthEstimate(distance_m=round(distance_m, 1), source="midas")


def get_depth_estimator(backend: str = config.DEPTH_BACKEND) -> DepthEstimator:
    """Factory with automatic, logged fallback to the geometric estimator."""
    if backend == "midas":
        try:
            estimator = MiDaSDepthEstimator()
            estimator._load()  # force load now so failures surface immediately
            return estimator
        except Exception as e:  # noqa: BLE001 - intentional broad fallback
            logger.warning(
                "MiDaS depth backend unavailable (%s). Falling back to "
                "geometric depth estimation.", e
            )
            return GeometricDepthEstimator()

    return GeometricDepthEstimator()
