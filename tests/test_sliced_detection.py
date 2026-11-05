from copy import deepcopy

from tools.evaluation.sliced_detection import (
    crowd_filter,
    deduplicate,
    restore_tile_boxes,
    tile_windows,
)


def box(bounds, confidence=0.8):
    return {"bbox": bounds, "confidence": confidence}


def test_restore_coordinates_and_reject_internal_fragments_without_mutation():
    detections = [box([0, 10, 20, 40]), box([10, 10, 30, 40]), box([30, 30, 60, 60])]
    original = deepcopy(detections)
    # Top and left are internal; bottom and right are real image edges.
    result = restore_tile_boxes(detections, (40, 40, 100, 100), (100, 100))
    assert result == [box([50, 50, 70, 80]), box([70, 70, 100, 100])]
    assert detections == original


def test_tiles_overlap_cover_image_and_skip_tiny_frames():
    windows = tile_windows(100, 200)
    assert len(windows) == 4
    assert windows[0] == (0, 0, 60, 120)
    assert windows[-1] == (40, 80, 100, 200)
    assert tile_windows(20, 40) == []


def test_merge_removes_duplicate_but_retains_neighbor_before_crowd_filter():
    duplicates = [box([0, 0, 10, 30], 0.2) for _ in range(30)]
    neighbor = box([8, 0, 18, 30], 0.8)
    merged = deduplicate([*duplicates, neighbor])
    assert len(merged) == 2
    # Thirty duplicate boxes must not trigger the crowded-scene confidence rule.
    assert crowd_filter(merged) == [neighbor]
    assert deduplicate([]) == []


def test_crowd_guard_uses_whole_frame_after_merge():
    boxes = [box([i * 20, 0, i * 20 + 10, 30], 0.2) for i in range(25)]
    assert len(crowd_filter(deduplicate(boxes))) == 25
    assert crowd_filter(boxes[:24]) == []
