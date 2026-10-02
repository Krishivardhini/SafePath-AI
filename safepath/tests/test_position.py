from safepath.core.position import classify_position, direction_phrase
from safepath.core.types import Position

FRAME_W = 640


def test_center_object_is_center():
    bbox = (280, 100, 360, 300)  # center_x = 320, dead center of 640
    assert classify_position(bbox, FRAME_W) == Position.CENTER


def test_left_object_is_left():
    bbox = (0, 100, 60, 300)  # center_x = 30
    assert classify_position(bbox, FRAME_W) == Position.LEFT


def test_right_object_is_right():
    bbox = (580, 100, 640, 300)  # center_x = 610
    assert classify_position(bbox, FRAME_W) == Position.RIGHT


def test_direction_phrase_ahead_for_center():
    bbox = (280, 100, 360, 300)
    assert direction_phrase(bbox, FRAME_W) == "ahead"


def test_direction_phrase_slightly_right():
    bbox = (400, 100, 460, 300)  # center_x = 430, moderately right of center
    phrase = direction_phrase(bbox, FRAME_W)
    assert phrase == "slightly right"


def test_direction_phrase_far_left():
    bbox = (0, 100, 20, 300)  # center_x = 10, far left edge
    phrase = direction_phrase(bbox, FRAME_W)
    assert phrase == "far left"
