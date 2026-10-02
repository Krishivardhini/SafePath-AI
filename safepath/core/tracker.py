"""Small per-camera IoU tracker for stable IDs and cautious motion labels."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time

from safepath.core.types import Detection


@dataclass
class _Track:
    track_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    distance_m: Optional[float]
    misses: int = 0
    centers: Tuple[Tuple[float, float], ...] = ()
    updated_at: float = 0.0


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(1, a[2] - a[0]) * max(1, a[3] - a[1])
    area_b = max(1, b[2] - b[0]) * max(1, b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


class IoUTracker:
    """Greedy same-class IoU matching; state must be scoped to one camera."""

    def __init__(self, max_misses=8):
        self.max_misses = max_misses
        self._next_id = 1
        self._tracks: Dict[int, _Track] = {}

    def update(self, detections: List[Detection], distances: List[Optional[float]]):
        now = time.monotonic()
        candidates = []
        for index, det in enumerate(detections):
            for track_id, track in self._tracks.items():
                if det.class_name == track.class_name:
                    overlap = _iou(det.bbox, track.bbox)
                    if overlap >= 0.12:
                        candidates.append((overlap, index, track_id))
        candidates.sort(reverse=True)
        matched_detections, matched_tracks = set(), set()
        assignments = {}
        for _, index, track_id in candidates:
            if index in matched_detections or track_id in matched_tracks:
                continue
            matched_detections.add(index)
            matched_tracks.add(track_id)
            assignments[index] = track_id

        motion = {}
        for index, det in enumerate(detections):
            distance = distances[index] if index < len(distances) else None
            if index not in assignments:
                track_id = self._next_id
                self._next_id += 1
                label = "unknown"
            else:
                track_id = assignments[index]
                previous = self._tracks[track_id]
                old_cx = (previous.bbox[0] + previous.bbox[2]) / 2.0
                new_cx = (det.bbox[0] + det.bbox[2]) / 2.0
                old_w = max(1, previous.bbox[2] - previous.bbox[0])
                old_h = max(1, previous.bbox[3] - previous.bbox[1])
                new_w = max(1, det.bbox[2] - det.bbox[0])
                new_h = max(1, det.bbox[3] - det.bbox[1])
                old_area = old_w * old_h
                new_area = new_w * new_h

                delta = None if distance is None or previous.distance_m is None else previous.distance_m - distance
                if delta is not None and delta > 0.20:
                    label = "approaching"
                elif delta is not None and delta < -0.20:
                    label = "moving away"
                elif new_area > old_area * 1.15:
                    label = "approaching"
                elif new_area < old_area * 0.85:
                    label = "moving away"
                elif abs(new_cx - old_cx) >= 14:
                    label = "crossing"
                else:
                    label = "stationary"

            det.track_id = track_id
            det.is_approaching = (label == "approaching")
            old_track = self._tracks.get(track_id)
            center = ((det.bbox[0] + det.bbox[2]) / 2.0, (det.bbox[1] + det.bbox[3]) / 2.0)
            centers = (old_track.centers if old_track else ()) + (center,)
            centers = centers[-15:]
            dt = max(now - old_track.updated_at, 0.001) if old_track and old_track.updated_at else 1.0
            det.velocity = ((center[0] - centers[-2][0]) / dt, (center[1] - centers[-2][1]) / dt) if len(centers) > 1 else (0.0, 0.0)
            motion[index] = label
            self._tracks[track_id] = _Track(track_id, det.class_name, det.bbox, distance, centers=centers, updated_at=now)

        for track_id in list(self._tracks):
            if track_id not in matched_tracks and track_id not in assignments.values():
                self._tracks[track_id].misses += 1
                if self._tracks[track_id].misses > self.max_misses:
                    del self._tracks[track_id]
        return motion
