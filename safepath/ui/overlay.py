"""
On-screen overlay. Note: the PRIMARY interface for a visually impaired
user is audio + haptic, not this window. This overlay exists for
development/debugging and for sighted companions/caregivers, so it's
designed to be high-contrast and legible at a glance, not decorative.
"""

from typing import List, Optional

import cv2
import numpy as np

from safepath.core.types import RiskLevel, ScoredDetection

_RISK_COLOR_BGR = {
    RiskLevel.LOW: (0, 200, 0),        # green
    RiskLevel.MEDIUM: (0, 200, 255),   # amber
    RiskLevel.HIGH: (0, 100, 255),     # orange
    RiskLevel.CRITICAL: (0, 0, 255),   # red
}

_DISCLAIMER = "Distances are APPROXIMATE. Assistive aid only -- not a safety guarantee."


def draw_overlay(
    frame: np.ndarray,
    scored_detections: List[ScoredDetection],
    top: Optional[ScoredDetection],
    spoken_message: str,
    fps: float,
) -> np.ndarray:
    frame = frame.copy()
    h, w = frame.shape[:2]

    for sd in scored_detections:
        x1, y1, x2, y2 = sd.detection.bbox
        color = _RISK_COLOR_BGR[sd.risk_level]
        thickness = 3 if sd is top else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        dist_str = f"{sd.distance_m:.1f}m~" if sd.distance_m is not None else "?"
        label = f"{sd.detection.class_name} {dist_str} [{sd.position.value}]"
        cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    # Bottom accessible caption bar: large, high-contrast, current warning.
    bar_height = 70
    cv2.rectangle(frame, (0, h - bar_height), (w, h), (0, 0, 0), -1)
    caption = spoken_message if spoken_message else "Path appears clear"
    cv2.putText(frame, caption, (12, h - bar_height + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # Top disclaimer strip, always visible.
    cv2.rectangle(frame, (0, 0), (w, 26), (0, 0, 0), -1)
    cv2.putText(frame, _DISCLAIMER, (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, h - bar_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return frame
