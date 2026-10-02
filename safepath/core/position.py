"""
Turns a bounding box into a coarse LEFT/CENTER/RIGHT position plus a
natural-language direction phrase for the voice warning.

Pure logic, no model dependency -- easy to unit test.
"""

from typing import Tuple

from safepath import config
from safepath.core.types import Position


def classify_position(
    bbox: Tuple[int, int, int, int],
    frame_width: int,
    center_zone_fraction: float = config.CENTER_ZONE_FRACTION,
) -> Position:
    """
    Classifies the horizontal position of an object's center point.

    The frame is split into a center zone (width = center_zone_fraction *
    frame_width) and two equal side zones.
    """
    x1, _, x2, _ = bbox
    center_x = (x1 + x2) / 2.0

    half_center = (center_zone_fraction * frame_width) / 2.0
    frame_center = frame_width / 2.0

    if frame_center - half_center <= center_x <= frame_center + half_center:
        return Position.CENTER
    if center_x < frame_center:
        return Position.LEFT
    return Position.RIGHT


def direction_phrase(bbox: Tuple[int, int, int, int], frame_width: int) -> str:
    """
    Produces a natural phrase like "slightly right", "far left", "ahead"
    used when building the spoken warning. Splits each side zone in half
    to distinguish "slightly" from "far".
    """
    x1, x2 = bbox[0], bbox[2]
    center_x = (x1 + x2) / 2.0
    offset_fraction = (center_x - frame_width / 2.0) / (frame_width / 2.0)  # -1..1

    position = classify_position(bbox, frame_width)
    if position == Position.CENTER:
        return "ahead"

    magnitude = abs(offset_fraction)
    qualifier = "far" if magnitude > 0.6 else "slightly"
    side = "left" if position == Position.LEFT else "right"
    return f"{qualifier} {side}"
