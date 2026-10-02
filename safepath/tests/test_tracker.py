from safepath.core.tracker import IoUTracker
from safepath.core.types import Detection


def test_tracker_keeps_id_and_marks_approaching():
    tracker = IoUTracker()
    first = Detection("person", "person", 0.9, (100, 80, 160, 250))
    assert tracker.update([first], [3.0])[0] == "unknown"
    first_id = first.track_id

    second = Detection("person", "person", 0.9, (102, 80, 162, 250))
    assert tracker.update([second], [2.5])[0] == "approaching"
    assert second.track_id == first_id
    assert second.velocity is not None


def test_tracker_retires_objects_after_misses():
    tracker = IoUTracker(max_misses=2)
    first = Detection("chair", "furniture", 0.9, (100, 80, 160, 160))
    tracker.update([first], [2.0])
    tracker.update([], [])
    tracker.update([], [])
    tracker.update([], [])
    next_object = Detection("chair", "furniture", 0.9, (101, 80, 161, 160))
    tracker.update([next_object], [2.0])
    assert next_object.track_id != first.track_id
