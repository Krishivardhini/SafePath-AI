from safepath.core.message_builder import build_warning
from safepath.core.risk_engine import build_scored_detection
from safepath.core.types import Detection, Position

FRAME_W = 640


def make_scored(class_name, category, bbox, distance_m, position):
    det = Detection(class_name=class_name, category=category, confidence=0.9, bbox=bbox)
    return build_scored_detection(det, distance_m, "geometric", position)


def test_person_location_message_does_not_claim_motion():
    scored = make_scored("person", "person", (0, 100, 60, 300), 1.5, Position.LEFT)
    msg = build_warning(scored, FRAME_W)
    assert msg == "Person, close, far left."


def test_generic_object_message_format():
    scored = make_scored("chair", "furniture", (280, 100, 360, 300), 1.2, Position.CENTER)
    msg = build_warning(scored, FRAME_W)
    assert msg == "Chair, close, ahead."


def test_very_close_distance_phrase():
    scored = make_scored("car", "vehicle", (280, 100, 360, 300), 0.8, Position.CENTER)
    msg = build_warning(scored, FRAME_W)
    assert "very close" in msg


def test_unknown_distance_omits_distance_phrase():
    scored = make_scored("bottle", "small_object", (400, 100, 460, 300), None, Position.RIGHT)
    msg = build_warning(scored, FRAME_W)
    assert msg == "Bottle, slightly right."
