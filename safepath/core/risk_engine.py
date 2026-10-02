"""
Combines distance, walking-path position, category, and confidence into a
single 0-100 risk score,
then selects the single most dangerous obstacle to warn about.

This is the heart of SafePath's "don't announce everything" principle:
`select_highest_risk` always returns at most one obstacle per frame.

Pure logic, no model dependency -- easy to unit test.
"""

from typing import List, Optional

from safepath import config
from safepath.core.types import Detection, Position, RiskLevel, ScoredDetection


def _distance_factor(distance_m: Optional[float]) -> float:
    """
    1.0 = right in front of the user, 0.0 = at/beyond MAX_RELEVANT_DISTANCE_M.
    Unknown distance gets a conservative mid-value so it's not ignored
    outright but also doesn't dominate known-close obstacles.
    """
    if distance_m is None:
        return 0.5
    if distance_m <= 0:
        return 1.0
    factor = 1.0 - (distance_m / config.MAX_RELEVANT_DISTANCE_M)
    return max(0.0, min(1.0, factor))


def _risk_level_from_score(score: float) -> RiskLevel:
    if score >= config.RISK_LEVEL_THRESHOLDS["CRITICAL"]:
        return RiskLevel.CRITICAL
    if score >= config.RISK_LEVEL_THRESHOLDS["HIGH"]:
        return RiskLevel.HIGH
    if score >= config.RISK_LEVEL_THRESHOLDS["MEDIUM"]:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def score_detection(
    detection: Detection,
    distance_m: Optional[float],
    position: Position,
) -> float:
    """Returns a risk score in [0, 100]."""
    # Distance and walking-path position determine whether an object is a
    # relevant obstacle. Category distinguishes the likely impact only after
    # that: a person is not automatically a high-risk detection.
    category_weight = config.CATEGORY_WEIGHT.get(detection.category, 0.5)
    position_weight = config.POSITION_WEIGHT.get(position.value, 0.7)
    dist_factor = _distance_factor(distance_m)

    # Confidence acts as a mild damper: low-confidence detections shouldn't
    # trigger high-urgency alerts as readily as confident ones.
    confidence_factor = 0.5 + 0.5 * detection.confidence  # in [0.5, 1.0]

    raw = category_weight * position_weight * dist_factor * confidence_factor
    return round(raw * 100, 1)


def build_scored_detection(
    detection: Detection,
    distance_m: Optional[float],
    depth_source: str,
    position: Position,
) -> ScoredDetection:
    score = score_detection(detection, distance_m, position)
    return ScoredDetection(
        detection=detection,
        distance_m=distance_m,
        depth_source=depth_source,
        position=position,
        risk_score=score,
        risk_level=_risk_level_from_score(score),
    )


def select_highest_risk(
    scored_detections: List[ScoredDetection],
) -> Optional[ScoredDetection]:
    """
    Returns the single most dangerous obstacle, or None if the list is
    empty. This is what enforces "warn about one thing, not everything".
    """
    if not scored_detections:
        return None
    return max(scored_detections, key=lambda sd: sd.risk_score)
