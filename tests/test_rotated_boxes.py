import math

import pytest

from src.bounding_box import BBFormat, BBType, BoundingBox
from src.evaluators.coco_evaluator import get_coco_metrics


def rotated(cx, cy, w, h, angle, detected=False, confidence=0.9):
    kwargs = dict(image_name='document', class_id='text',
                  coordinates=(cx, cy, w, h, angle), format=BBFormat.CXCYWHA)
    if detected:
        kwargs.update(bb_type=BBType.DETECTED, confidence=confidence)
    return BoundingBox(**kwargs)


def test_equivalent_horizontal_and_width_height_swap():
    horizontal = rotated(0, 0, 4, 1, 0)
    swapped = rotated(0, 0, 1, 4, 90)

    assert horizontal.get_continuous_area() == pytest.approx(4.0)
    assert BoundingBox.get_intersection_area(horizontal, swapped) == pytest.approx(4.0)
    assert BoundingBox.iou(horizontal, horizontal) == pytest.approx(1.0)
    assert BoundingBox.iou(horizontal, swapped) == pytest.approx(1.0)


def test_mirrored_slender_boxes_have_real_iou_one_seventh():
    clockwise = rotated(0, 0, 4, 1, 45, detected=True)
    counter_clockwise = rotated(0, 0, 4, 1, -45)

    inter = BoundingBox.get_intersection_area(clockwise, counter_clockwise)
    iou = BoundingBox.iou(clockwise, counter_clockwise)

    assert inter == pytest.approx(1.0, abs=1e-9)
    assert iou == pytest.approx(1.0 / 7.0, abs=1e-9)


def test_mirrored_pair_is_not_tp_at_half_threshold():
    det = rotated(0, 0, 4, 1, 45, detected=True)
    gt = rotated(0, 0, 4, 1, -45)

    result = get_coco_metrics([gt], [det], iou_threshold=0.5)['text']

    assert result['TP'] == 0
    assert result['FP'] == 1


def test_rotated_and_ordinary_boxes_match_in_same_evaluation():
    rotated_horizontal = rotated(0, 0, 4, 1, 0, detected=True)
    # Ordinary XYWH gives upper-left (-2, -0.5), hence the same rectangle.
    ordinary_gt = BoundingBox('document', 'text', (-2, -0.5, 4, 1))

    assert BoundingBox.iou(rotated_horizontal, ordinary_gt) == pytest.approx(1.0)
    result = get_coco_metrics([ordinary_gt], [rotated_horizontal], 0.5)['text']
    assert result['TP'] == 1
    assert result['FP'] == 0


def test_degenerate_box_does_not_divide_by_zero_or_match():
    normal = BoundingBox('document', 'text', (-2, -0.5, 4, 1))
    line = BoundingBox('document', 'text', (0, 0, 0, 1))

    assert BoundingBox.continuous_iou(normal, line) == 0.0
    assert BoundingBox.continuous_iou(line, line) == 0.0
