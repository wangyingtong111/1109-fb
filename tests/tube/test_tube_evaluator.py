import json
import os

import numpy as np
import pytest

from src.evaluators.tube_evaluator import TubeEvaluator
from src.tube import Tube
from src.utils.enumerators import TubeMatchingMethod


# TODO: More tests!!!
def test_tube_eval():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    example_anno = os.path.join(this_dir, 'example_anno.json')
    example_preds = os.path.join(this_dir, 'example_preds.json')

    tube_evaluator = TubeEvaluator(example_anno, example_preds)
    res, mAP = tube_evaluator.evaluate(thr=0.5)

    assert mAP == 1.0


def _tube(bbox, confidence=None):
    """Build a one-frame tube (bbox is xywh and gets converted to xyxy internally).
    No 'id' attribute is set, so the evaluator falls back to the list position."""
    det = {'frame': 0, 'bbox': bbox}
    if confidence is not None:
        det['confidence'] = confidence
    return Tube(category_id=0, video_id=0, track=[det])


def _make_evaluator():
    # the constructor only validates the extensions; files are read inside evaluate()
    return TubeEvaluator('anno.json', 'preds.json')


def test_confidence_first_high_confidence_gets_tp_and_duplicate_is_fp():
    """Reference scenario: one gt, high-conf pred (.9) with IoU=.6, low-conf pred (.4) with
    IoU=1.0, threshold .5. The high-confidence prediction wins the only gt (TP) and the
    perfect-overlap but lower-confidence prediction becomes a duplicate FP."""
    gt = _tube([0, 0, 10, 10])
    high = _tube([2.5, 0, 10, 10], confidence=0.9)  # IoU = 0.6
    low = _tube([0, 0, 10, 10], confidence=0.4)     # IoU = 1.0

    evaluator = _make_evaluator()
    n_tp, n_fp, n_fn, pairings = evaluator._classify_tubes_confidence_first(
        [high, low], [gt], thr=0.5)

    assert (n_tp, n_fp, n_fn) == (1, 1, 0)
    assert high.isTP is True
    assert low.isTP is False
    assert high.matched_gt_id is not None
    assert low.matched_gt_id is None

    by_pred = {p['pred_id']: p for p in pairings}
    # ids fall back to the position in the input list
    assert by_pred[0]['status'] == 'TP'
    assert by_pred[0]['iou'] == pytest.approx(0.6)
    assert by_pred[1]['status'] == 'duplicate FP'
    assert by_pred[1]['iou'] == pytest.approx(1.0)
    assert by_pred[1]['gt_id'] is None


def test_confidence_first_tie_breaking_is_reproducible_by_input_order():
    """Two predictions with the same confidence: the one appearing first in the input must
    deterministically get the gt, regardless of their IoU."""
    gt = _tube([0, 0, 10, 10])
    # both confidence .5; first one has the smaller IoU
    first = _tube([2.5, 0, 10, 10], confidence=0.5)  # IoU = 0.6
    second = _tube([0, 0, 10, 10], confidence=0.5)   # IoU = 1.0

    evaluator = _make_evaluator()
    _, _, _, pairings_a = evaluator._classify_tubes_confidence_first(
        [first, second], [gt], thr=0.5)
    _, _, _, pairings_b = evaluator._classify_tubes_confidence_first(
        [first, second], [gt], thr=0.5)

    assert pairings_a == pairings_b  # reproducible
    statuses = [p['status'] for p in pairings_a]
    assert statuses == ['TP', 'duplicate FP']

    # swapping the input order swaps which prediction gets the TP
    _, _, _, pairings_swapped = evaluator._classify_tubes_confidence_first(
        [second, first], [gt], thr=0.5)
    assert [p['status'] for p in pairings_swapped] == ['TP', 'duplicate FP']


def test_confidence_first_each_prediction_used_once():
    """Two gts and three predictions: every prediction produces exactly one pairing and no
    prediction is matched twice; each gt can be matched at most once."""
    gt1 = _tube([0, 0, 10, 10])
    gt2 = _tube([100, 0, 10, 10])
    p1 = _tube([0, 0, 10, 10], confidence=0.9)    # IoU 1.0 with gt1
    p2 = _tube([100, 0, 10, 10], confidence=0.8)  # IoU 1.0 with gt2
    p3 = _tube([0, 0, 10, 10], confidence=0.7)    # duplicate of gt1

    evaluator = _make_evaluator()
    n_tp, n_fp, n_fn, pairings = evaluator._classify_tubes_confidence_first(
        [p1, p2, p3], [gt1, gt2], thr=0.5)

    assert (n_tp, n_fp, n_fn) == (2, 1, 0)
    assert len(pairings) == 3  # one entry per prediction
    matched_gts = [p['gt_id'] for p in pairings if p['status'] == 'TP']
    assert len(matched_gts) == len(set(matched_gts)) == 2  # gt matched at most once
    assert [p['status'] for p in pairings] == ['TP', 'TP', 'duplicate FP']


def test_max_iou_strategy_still_available_and_reports_pairings():
    """The legacy maximum-overlap-first association must remain callable. Here it gives the
    perfect-overlap (but low-confidence) prediction the TP, showing it ignores confidence."""
    gt = _tube([0, 0, 10, 10])
    high = _tube([2.5, 0, 10, 10], confidence=0.9)  # IoU = 0.6
    low = _tube([0, 0, 10, 10], confidence=0.4)     # IoU = 1.0

    evaluator = _make_evaluator()
    n_tp, n_fp, n_fn, pairings = evaluator._classify_tubes_max_iou(
        [high, low], [gt], thr=0.5)

    assert (n_tp, n_fp, n_fn) == (1, 1, 0)
    assert high.isTP is False
    assert low.isTP is True  # best overlap wins under the legacy strategy
    by_pred = {p['pred_id']: p for p in pairings}
    assert by_pred[0]['status'] == 'duplicate FP'
    assert by_pred[1]['status'] == 'TP'
    assert by_pred[1]['iou'] == pytest.approx(1.0)


def _write_json_dataset(tmp_path):
    anno = {
        'videos': [{'file_name': 'v1.mp4', 'width': 100, 'height': 100, 'id': 0}],
        'annotations': [{
            'id': 0, 'video_id': 0, 'category_id': 0,
            'track': [{'frame': 0, 'bbox': [0, 0, 10, 10]}],
        }],
        'categories': [{'id': 0, 'name': 'obj'}],
    }
    preds = [
        # high confidence first, lower IoU
        {'id': 0, 'video_id': 0, 'category_id': 0,
         'track': [{'frame': 0, 'bbox': [2.5, 0, 10, 10], 'confidence': 0.9}]},
        # low confidence, perfect IoU
        {'id': 1, 'video_id': 0, 'category_id': 0,
         'track': [{'frame': 0, 'bbox': [0, 0, 10, 10], 'confidence': 0.4}]},
    ]
    anno_path = os.path.join(tmp_path, 'anno.json')
    preds_path = os.path.join(tmp_path, 'preds.json')
    with open(anno_path, 'w') as f:
        json.dump(anno, f)
    with open(preds_path, 'w') as f:
        json.dump(preds, f)
    return anno_path, preds_path


def test_evaluate_confidence_first_matches_pr_order(tmp_path):
    """End-to-end: under confidence-first association the PR arrays follow the confidence
    order, i.e. the first (highest confidence) prediction is the TP."""
    anno_path, preds_path = _write_json_dataset(tmp_path)

    evaluator = TubeEvaluator(anno_path, preds_path,
                              matching_method=TubeMatchingMethod.CONFIDENCE)
    res, mAP = evaluator.evaluate(thr=0.5)

    # predictions sorted by decreasing confidence: high (.9) then low (.4)
    tp = [int(tube.isTP) for tube in evaluator._predictions]
    assert tp == [1, 0]  # PR order and association order agree

    result = res['obj']
    assert result['total TP'] == 1
    assert result['total FP'] == 1
    assert result['total FN'] == 0
    assert result['matching method'] == TubeMatchingMethod.CONFIDENCE
    statuses = [p['status'] for p in evaluator.get_pairings()['obj']]
    assert statuses == ['TP', 'duplicate FP']
    # precision after the TP then after the duplicate FP
    assert np.array_equal(result['precision'], np.array([1.0, 0.5]))


def test_evaluate_defaults_to_legacy_max_iou(tmp_path):
    """Without specifying a matching method, the legacy behaviour is kept."""
    anno_path, preds_path = _write_json_dataset(tmp_path)

    evaluator = TubeEvaluator(anno_path, preds_path)
    res, _ = evaluator.evaluate(thr=0.5)

    assert res['obj']['matching method'] == TubeMatchingMethod.MAX_IOU
    statuses = [p['status'] for p in evaluator.get_pairings()['obj']]
    assert statuses == ['duplicate FP', 'TP']

    # the matching method can also be overridden per evaluate() call
    res_override, _ = evaluator.evaluate(thr=0.5,
                                         matching_method=TubeMatchingMethod.CONFIDENCE)
    assert res_override['obj']['matching method'] == TubeMatchingMethod.CONFIDENCE


def test_evaluate_rejects_invalid_matching_method(tmp_path):
    anno_path, preds_path = _write_json_dataset(tmp_path)
    evaluator = TubeEvaluator(anno_path, preds_path)
    with pytest.raises(ValueError):
        evaluator.evaluate(thr=0.5, matching_method='not_a_method')
