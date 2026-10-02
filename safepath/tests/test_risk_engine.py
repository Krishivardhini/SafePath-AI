from safepath.core.risk_engine import (
    score_detection, build_scored_detection, select_highest_risk,
)
from safepath.core.types import Detection, Position, RiskLevel


def make_detection(class_name="person", category="person", confidence=0.9,
                    bbox=(280, 100, 360, 300)):
    return Detection(class_name=class_name, category=category,
                      confidence=confidence, bbox=bbox)


def test_closer_object_scores_higher():
    det = make_detection()
    near = score_detection(det, distance_m=0.5, position=Position.CENTER)
    far = score_detection(det, distance_m=5.0, position=Position.CENTER)
    assert near > far


def test_center_scores_higher_than_side_at_same_distance():
    det = make_detection()
    center = score_detection(det, distance_m=1.0, position=Position.CENTER)
    left = score_detection(det, distance_m=1.0, position=Position.LEFT)
    assert center > left


def test_vehicle_outranks_small_object_at_same_distance_and_position():
    vehicle = make_detection(class_name="car", category="vehicle")
    bottle = make_detection(class_name="bottle", category="small_object")
    vehicle_score = score_detection(vehicle, distance_m=2.0, position=Position.CENTER)
    bottle_score = score_detection(bottle, distance_m=2.0, position=Position.CENTER)
    assert vehicle_score > bottle_score


def test_person_beside_user_is_low_risk_even_when_close():
    det = make_detection()
    scored = build_scored_detection(det, distance_m=0.3, depth_source="geometric",
                                     position=Position.LEFT)
    assert scored.risk_level == RiskLevel.LOW


def test_person_far_away_is_low_risk():
    scored = build_scored_detection(
        make_detection(), distance_m=5.0, depth_source="geometric", position=Position.CENTER
    )
    assert scored.risk_level == RiskLevel.LOW


def test_person_directly_ahead_but_far_is_not_high_risk():
    scored = build_scored_detection(
        make_detection(), distance_m=3.0, depth_source="geometric", position=Position.CENTER
    )
    assert scored.risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM)


def test_person_directly_ahead_and_very_close_is_high_risk():
    scored = build_scored_detection(
        make_detection(), distance_m=0.3, depth_source="geometric", position=Position.CENTER
    )
    assert scored.risk_level == RiskLevel.HIGH


def test_bicycle_directly_ahead_is_high_risk():
    scored = build_scored_detection(
        make_detection(class_name="bicycle", category="vehicle"),
        distance_m=2.0,
        depth_source="geometric",
        position=Position.CENTER,
    )
    assert scored.risk_level == RiskLevel.HIGH


def test_car_directly_ahead_is_high_risk():
    scored = build_scored_detection(
        make_detection(class_name="car", category="vehicle"),
        distance_m=2.0,
        depth_source="geometric",
        position=Position.CENTER,
    )
    assert scored.risk_level == RiskLevel.HIGH


def test_object_outside_walking_path_has_lower_priority():
    det = make_detection(class_name="car", category="vehicle")
    ahead = score_detection(det, distance_m=2.0, position=Position.CENTER)
    beside = score_detection(det, distance_m=2.0, position=Position.RIGHT)
    assert beside < ahead
    assert build_scored_detection(
        det, distance_m=2.0, depth_source="geometric", position=Position.RIGHT
    ).risk_level == RiskLevel.LOW


def test_far_low_priority_object_is_low_risk():
    det = make_detection(class_name="bottle", category="small_object")
    scored = build_scored_detection(det, distance_m=5.5, depth_source="geometric",
                                     position=Position.RIGHT)
    assert scored.risk_level == RiskLevel.LOW


def test_select_highest_risk_picks_most_dangerous_only():
    near_person = build_scored_detection(
        make_detection(), distance_m=0.5, depth_source="geometric", position=Position.CENTER
    )
    far_bottle = build_scored_detection(
        make_detection(class_name="bottle", category="small_object"),
        distance_m=4.0, depth_source="geometric", position=Position.LEFT
    )
    top = select_highest_risk([near_person, far_bottle])
    assert top is near_person


def test_select_highest_risk_prefers_relevant_path_obstacle_over_person_beside_user():
    person_beside = build_scored_detection(
        make_detection(), distance_m=0.3, depth_source="geometric", position=Position.LEFT
    )
    bicycle_ahead = build_scored_detection(
        make_detection(class_name="bicycle", category="vehicle"),
        distance_m=2.0,
        depth_source="geometric",
        position=Position.CENTER,
    )
    assert select_highest_risk([person_beside, bicycle_ahead]) is bicycle_ahead


def test_select_highest_risk_empty_list_returns_none():
    assert select_highest_risk([]) is None
