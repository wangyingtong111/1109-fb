import os

import pytest

from src.evaluators.tube_evaluator import TubeEvaluator
from src.tube import Tube


# TODO: More tests!!!
def test_tube_eval():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    example_anno = os.path.join(this_dir, 'example_anno.json')
    example_preds = os.path.join(this_dir, 'example_preds.json')

    tube_evaluator = TubeEvaluator(example_anno, example_preds)
    res, mAP = tube_evaluator.evaluate(thr=0.5)

    assert mAP == 1.0


def test_tube_iou_with_overlapping_boxes_in_same_frame():
    # prediction has two overlapping xywh boxes in the same frame;
    # the ground truth is a single box covering both
    pred = Tube(category_id=0, video_id=0, track=[
        {'frame': 0, 'bbox': [0, 0, 2, 2], 'confidence': 0.9},
        {'frame': 0, 'bbox': [1, 0, 2, 2], 'confidence': 0.8},
    ])
    gt = Tube(category_id=0, video_id=0, track=[
        {'frame': 0, 'bbox': [0, 0, 3, 2]},
    ])

    evaluator = TubeEvaluator.__new__(TubeEvaluator)

    # the union of the predicted boxes is exactly the gt box: area 6
    assert pred.get_tube_volume() == pytest.approx(6.0)
    assert gt.get_tube_volume() == pytest.approx(6.0)
    assert evaluator._tubes_inter(pred, gt) == pytest.approx(6.0)

    iou = evaluator._tube_pairwise_iou([pred], [gt])
    assert iou.shape == (1, 1)
    assert iou[0, 0] == pytest.approx(1.0)


def test_tube_iou_with_different_box_counts_per_frame_in_same_batch():
    # tubes whose frames have different numbers of boxes, evaluated in one batch
    pred = Tube(category_id=0, video_id=0, track=[
        {'frame': 0, 'bbox': [0, 0, 2, 2], 'confidence': 0.9},
        {'frame': 0, 'bbox': [1, 0, 2, 2], 'confidence': 0.8},
        {'frame': 1, 'bbox': [0, 0, 3, 2], 'confidence': 0.7},
    ])
    gt1 = Tube(category_id=0, video_id=0, track=[
        {'frame': 0, 'bbox': [0, 0, 3, 2]},
        {'frame': 1, 'bbox': [0, 0, 1, 2]},
        {'frame': 1, 'bbox': [2, 0, 1, 2]},
    ])
    gt2 = Tube(category_id=0, video_id=0, track=[
        {'frame': 0, 'bbox': [0, 0, 3, 2]},
        {'frame': 1, 'bbox': [0, 0, 3, 2]},
    ])

    evaluator = TubeEvaluator.__new__(TubeEvaluator)

    # pred volume: frame 0 -> union of overlapping boxes = 6; frame 1 -> 6
    assert pred.get_tube_volume() == pytest.approx(12.0)
    # gt1 volume: frame 0 -> 6; frame 1 -> two disjoint boxes = 4
    assert gt1.get_tube_volume() == pytest.approx(10.0)
    # gt2 volume: 6 + 6
    assert gt2.get_tube_volume() == pytest.approx(12.0)

    iou = evaluator._tube_pairwise_iou([pred], [gt1, gt2])
    assert iou.shape == (1, 2)
    # pred x gt1: inter = 6 (frame 0) + 4 (frame 1) = 10; union = 12 + 10 - 10
    assert iou[0, 0] == pytest.approx(10.0 / 12.0)
    # pred x gt2: inter = 6 + 6 = 12; union = 12 + 12 - 12
    assert iou[0, 1] == pytest.approx(1.0)
