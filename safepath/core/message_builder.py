"""
Turns a ScoredDetection into a short, actionable phrase, following the
project's core principle: one obstacle, one short sentence.

Pure logic, no model dependency -- easy to unit test.
"""

from safepath.core.position import direction_phrase
from safepath.core.types import ScoredDetection

_DISPLAY_NAME = {
    "person": "Person",
    "bicycle": "Bicycle",
    "car": "Car",
    "motorcycle": "Motorcycle",
    "bus": "Bus",
    "truck": "Truck",
    "chair": "Chair",
    "backpack": "Backpack",
    "handbag": "Bag",
    "suitcase": "Suitcase",
    "bottle": "Bottle",
    "bench": "Bench",
    "potted plant": "Plant",
    "dog": "Dog",
}


def _distance_phrase(distance_m):
    if distance_m is None:
        return None
    if distance_m < 1.0:
        return "very close"
    if distance_m < 2.0:
        return "close"
    # Round to nearest 0.5m and always phrase as approximate.
    rounded = round(distance_m * 2) / 2
    return f"about {rounded:g} meters"


def build_warning(scored: ScoredDetection, frame_width: int) -> str:
    name = _DISPLAY_NAME.get(scored.detection.class_name, scored.detection.class_name.title())
    direction = direction_phrase(scored.detection.bbox, frame_width)
    distance_phrase = _distance_phrase(scored.distance_m)

    parts = [name]
    if distance_phrase:
        parts.append(distance_phrase)
    parts.append(direction)
    return ", ".join(parts) + "."
