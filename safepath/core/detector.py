"""
SafePath Real-Time YOLOv8 Detection Engine.

High-throughput, optimized computer vision stage supporting:
- Lazy loading & automated device selection (CUDA / Apple Silicon MPS / CPU multi-threading).
- Half-precision (FP16) inference acceleration on CUDA GPUs.
- Zero-latency warm-up cycle on initialization to eliminate cold-start lag.
- O(1) class ID lookup cache with known safety categories and full class exposure.
- Granular per-stage telemetry profiling (preprocess, neural inference, NMS postprocess).
- Batch and single-frame inference APIs with robust input validation.
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from safepath import config
from safepath.core.types import Detection, DetectionResult
from safepath.utils.logger import get_logger

logger = get_logger(__name__)


class ObstacleDetector:
    """
    YOLOv8 detection pipeline with all model classes and downstream risk labels.
    Swappable with any backend conforming to `detect(frame) -> List[Detection]`.
    """

    def __init__(
        self,
        model_path: str = config.YOLO_MODEL_PATH,
        confidence: float = config.CONFIDENCE_THRESHOLD,
        iou: float = config.IOU_THRESHOLD,
        device: str = config.DETECTOR_DEVICE,
        img_size: int = config.INFERENCE_IMG_SIZE,
        half: bool = config.HALF_PRECISION,
        warmup: bool = config.WARMUP_ON_LOAD,
    ):
        self._model_path = self._resolve_model_path(model_path)
        self._confidence = confidence
        self._iou = iou
        self._requested_device = device
        self._img_size = img_size
        self._half = half
        self._warmup = warmup

        self._model = None
        self._device: str = "cpu"
        self._use_half: bool = False
        self._class_mapping: Dict[int, Tuple[str, str]] = {}
        self.last_telemetry: Dict[str, float] = {
            "preprocess_ms": 0.0,
            "inference_ms": 0.0,
            "postprocess_ms": 0.0,
            "total_ms": 0.0,
        }

    @staticmethod
    def _resolve_model_path(path: str) -> str:
        """Resolves model path across package directories and project root."""
        if os.path.isabs(path) and os.path.exists(path):
            return path

        candidates = [
            path,
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), path),
            os.path.join(os.getcwd(), path),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return os.path.abspath(candidate)

        # Default fallback to filename so ultralytics auto-downloads if needed
        return path

    def _determine_device(self) -> str:
        """Selects the optimal compute device based on hardware availability."""
        req = (self._requested_device or "auto").lower()
        if req in ("cuda", "0") and torch.cuda.is_available():
            return "cuda"
        if req == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if req == "cpu":
            return "cpu"

        # Auto-detect
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        # Optimize CPU threads for real-time edge processing
        try:
            threads = min(4, max(1, os.cpu_count() or 1))
            torch.set_num_threads(threads)
        except Exception:
            pass
        return "cpu"

    def _load(self):
        """Lazy-loads the YOLO model, configures compute device, and runs warm-up."""
        if self._model is not None:
            return

        t_start = time.perf_counter()
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is not installed. Run: pip install ultralytics"
            ) from e

        self._device = self._determine_device()
        self._use_half = bool(self._half and self._device == "cuda")

        logger.info(
            "Loading YOLO model '%s' on device '%s' (half_precision=%s)...",
            self._model_path,
            self._device,
            self._use_half,
        )

        self._model = YOLO(self._model_path)
        
        # Detect every class supplied by this model. Safety can weight known
        # hazards differently, but Object Explorer remains complete.
        names = self._model.names or {}
        self._class_mapping.clear()
        for class_id, class_name in names.items():
            clean_name = str(class_name).strip().lower()
            category = config.TARGET_CLASSES.get(clean_name, "other")
            self._class_mapping[int(class_id)] = (clean_name, category)

        load_time_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "YOLO model loaded in %.1f ms. Monitoring all %d model classes.",
            load_time_ms,
            len(self._class_mapping),
        )

        # Warm-up pass to eliminate initial inference stutter
        if self._warmup:
            self._run_warmup()

    def _predict_raw(self, source):
        kwargs = {
            "source": source,
            "conf": self._confidence,
            "iou": self._iou,
            "imgsz": self._img_size,
            "device": self._device,
            "verbose": False,
        }
        if self._use_half:
            kwargs["half"] = True
        return self._model.predict(**kwargs)

    def _run_warmup(self):
        """Executes a dummy inference pass to compile and cache kernel execution graphs."""
        try:
            t0 = time.perf_counter()
            dummy_frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
            self._predict_raw(dummy_frame)
            warmup_ms = (time.perf_counter() - t0) * 1000
            logger.info("Detector warm-up completed in %.1f ms.", warmup_ms)
        except Exception as e:
            logger.warning("Detector warm-up skipped due to: %s", e)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Runs object detection on a single BGR image.
        Returns filtered list of SafePath target obstacle detections.
        """
        result = self.detect_with_telemetry(frame)
        return result.detections

    def detect_with_telemetry(self, frame: np.ndarray) -> DetectionResult:
        """
        Executes real-time inference and returns both detections and timing telemetry.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return DetectionResult(detections=[])

        h, w = frame.shape[:2]
        if h < 10 or w < 10:
            return DetectionResult(detections=[])

        self._load()

        t0 = time.perf_counter()
        results = self._predict_raw(frame)
        t_total = (time.perf_counter() - t0) * 1000

        detections: List[Detection] = []
        if not results:
            return DetectionResult(detections=[], total_ms=round(t_total, 2), device=self._device, frame_shape=(h, w))

        res = results[0]
        now_ts = time.time()

        # Extract timing metrics from Ultralytics speed dict if available
        speed_dict = getattr(res, "speed", {}) or {}
        t_prep = float(speed_dict.get("preprocess", 0.0))
        t_inf = float(speed_dict.get("inference", t_total))
        t_post = float(speed_dict.get("postprocess", 0.0))

        self.last_telemetry = {
            "preprocess_ms": round(t_prep, 2),
            "inference_ms": round(t_inf, 2),
            "postprocess_ms": round(t_post, 2),
            "total_ms": round(t_total, 2),
        }

        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return DetectionResult(
                detections=[],
                preprocess_ms=t_prep,
                inference_ms=t_inf,
                postprocess_ms=t_post,
                total_ms=round(t_total, 2),
                device=self._device,
                frame_shape=(h, w),
            )

        # Vectorized / fast extraction
        classes = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        coords = boxes.xyxy.cpu().numpy().astype(int)

        for i in range(len(classes)):
            cid = classes[i]
            if cid not in self._class_mapping:
                continue

            class_name, category = self._class_mapping[cid]
            conf = float(confidences[i])
            x1, y1, x2, y2 = coords[i]

            # Clip bounding boxes to valid image coordinates
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            detections.append(
                Detection(
                    class_name=class_name,
                    category=category,
                    confidence=round(conf, 4),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    class_id=int(cid),
                    timestamp=now_ts,
                )
            )

        return DetectionResult(
            detections=detections,
            preprocess_ms=t_prep,
            inference_ms=t_inf,
            postprocess_ms=t_post,
            total_ms=round(t_total, 2),
            device=self._device,
            frame_shape=(h, w),
        )

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """
        Batched inference across multiple frames for multi-camera or offline review.
        """
        if not frames:
            return []

        self._load()
        valid_frames = [f for f in frames if f is not None and f.size > 0]
        if not valid_frames:
            return [[] for _ in frames]

        results = self._predict_raw(valid_frames)

        all_detections: List[List[Detection]] = []
        now_ts = time.time()

        for res, frame in zip(results, valid_frames):
            frame_dets: List[Detection] = []
            h, w = frame.shape[:2]
            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                classes = boxes.cls.cpu().numpy().astype(int)
                confidences = boxes.conf.cpu().numpy()
                coords = boxes.xyxy.cpu().numpy().astype(int)

                for i in range(len(classes)):
                    cid = classes[i]
                    if cid in self._class_mapping:
                        class_name, category = self._class_mapping[cid]
                        x1, y1, x2, y2 = coords[i]
                        x1 = max(0, min(x1, w - 1))
                        y1 = max(0, min(y1, h - 1))
                        x2 = max(x1 + 1, min(x2, w))
                        y2 = max(y1 + 1, min(y2, h))

                        frame_dets.append(
                            Detection(
                                class_name=class_name,
                                category=category,
                                confidence=round(float(confidences[i]), 4),
                                bbox=(int(x1), int(y1), int(x2), int(y2)),
                                class_id=int(cid),
                                timestamp=now_ts,
                            )
                        )
            all_detections.append(frame_dets)

        return all_detections
