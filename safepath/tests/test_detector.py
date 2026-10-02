"""
Unit tests for the SafePath Real-Time YOLOv8 Detection Engine.
"""

import numpy as np
import pytest

from safepath import config
from safepath.core.detector import ObstacleDetector
from safepath.core.types import Detection, DetectionResult


def test_detector_initialization():
    detector = ObstacleDetector(confidence=0.5, iou=0.4)
    assert detector._confidence == 0.5
    assert detector._iou == 0.4
    assert detector._model is None  # Lazy loading verification


def test_detector_device_selection():
    detector = ObstacleDetector(device="cpu")
    selected = detector._determine_device()
    assert selected == "cpu"


def test_detector_empty_and_corrupt_frames():
    detector = ObstacleDetector()
    
    # None frame
    res_none = detector.detect_with_telemetry(None)
    assert isinstance(res_none, DetectionResult)
    assert len(res_none.detections) == 0

    # Zero-sized numpy array
    res_empty = detector.detect_with_telemetry(np.array([], dtype=np.uint8))
    assert len(res_empty.detections) == 0

    # Too-small image (e.g. 4x4 pixels)
    res_tiny = detector.detect_with_telemetry(np.zeros((4, 4, 3), dtype=np.uint8))
    assert len(res_tiny.detections) == 0


def test_detector_inference_on_synthetic_frame():
    detector = ObstacleDetector(confidence=0.25, warmup=True)
    
    # Create a 640x480 synthetic frame (blank / noise)
    frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
    
    result = detector.detect_with_telemetry(frame)
    assert isinstance(result, DetectionResult)
    assert isinstance(result.detections, list)
    assert result.device in ("cpu", "cuda", "mps")
    assert result.total_ms >= 0.0
    assert result.preprocess_ms >= 0.0
    assert result.inference_ms >= 0.0
    assert result.postprocess_ms >= 0.0


def test_detector_batch_inference():
    detector = ObstacleDetector(confidence=0.3)
    frame1 = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
    frame2 = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
    
    batch_results = detector.detect_batch([frame1, frame2])
    assert len(batch_results) == 2
    assert isinstance(batch_results[0], list)
    assert isinstance(batch_results[1], list)


def test_all_model_classes_are_available_to_explorer():
    detector = ObstacleDetector()
    detector._load()
    
    # Safety categories are assigned to known hazards; all other pretrained
    # COCO classes remain visible to Object Explorer as "other".
    for cid, (cname, cat) in detector._class_mapping.items():
        assert cat == config.TARGET_CLASSES.get(cname, "other")
    assert any(name not in config.TARGET_CLASSES for name, _ in detector._class_mapping.values())
