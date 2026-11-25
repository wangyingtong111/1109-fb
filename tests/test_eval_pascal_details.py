################################################################################
# Tests for the optional per-detection details of the PASCAL VOC metrics.
#
# The details explain, for each detection, why it ended up as a TP, duplicate,
# poor localization, wrong class or ignored box, following the matching run
# being executed. The original AP / matching / sorting outputs must be
# unaffected, both when the details are requested and when they are not.
################################################################################

from math import isclose

from src.bounding_box import BoundingBox
from src.evaluators.pascal_voc_evaluator import get_pascalvoc_metrics
from src.utils.enumerators import (BBFormat, BBType, DetectionOutcome,
                                   MethodAveragePrecision)

IMG = 'img'


def _gt(class_id, coords):
    return BoundingBox(IMG,
                       class_id=class_id,
                       coordinates=coords,
                       bb_type=BBType.GROUND_TRUTH,
                       format=BBFormat.XYX2Y2)


def _det(class_id, confidence, coords, is_ignore=False):
    return BoundingBox(IMG,
                       class_id=class_id,
                       coordinates=coords,
                       bb_type=BBType.DETECTED,
                       confidence=confidence,
                       format=BBFormat.XYX2Y2,
                       is_ignore=is_ignore)


def _records(details, outcome):
    return details[outcome]


def _build_boxes():
    # Same-class ground truth A1: fully overlapped by the .9 detection, and at
    # IoU .75 with the .8 detection.
    gt_a1 = _gt('A', (0, 0, 99, 99))
    # Another same-class ground truth A2, not overlapped by the .8 detection.
    gt_a2 = _gt('A', (25, 100, 124, 199))
    # Ground truth of another class B: IoU 1 with the .8 detection.
    gt_b = _gt('B', (25, 0, 100, 99))

    det_90 = _det('A', 0.9, (0, 0, 99, 99))
    det_80 = _det('A', 0.8, (25, 0, 100, 99))
    return [gt_a1, gt_a2, gt_b], [det_90, det_80]


def test_tp_detail_points_to_actually_hit_ground_truth():
    gts, dets = _build_boxes()
    results = get_pascalvoc_metrics(gts, dets, iou_threshold=0.5, generate_detail=True)
    details = results['details']

    tp_records = _records(details, DetectionOutcome.TP)
    assert len(tp_records) == 1
    record = tp_records[0]
    # The .9 prediction fully overlaps its same-class ground truth...
    assert record['detection'].get_confidence() == 0.9
    # ...and the detail must locate the ground truth actually hit.
    assert record['matched_gt'] is gts[0]
    assert isclose(record['iou'], 1.0)


def test_duplicate_at_low_threshold_points_to_higher_confidence_detection():
    gts, dets = _build_boxes()
    results = get_pascalvoc_metrics(gts, dets, iou_threshold=0.5, generate_detail=True)
    details = results['details']

    # At .5 the .8 detection is a duplicate of the same-class GT already taken by .9.
    dup_records = _records(details, DetectionOutcome.DUPLICATE)
    assert len(dup_records) == 1
    record = dup_records[0]
    assert record['detection'].get_confidence() == 0.8
    assert record['matched_gt'] is gts[0]
    assert record['duplicate_of'] is dets[0]  # points to the .9 detection
    # Same-class IoU is around .75 (exact .75 is not representable with integer pixels).
    assert isclose(record['iou'], 0.75, abs_tol=1e-2)
    # No wrong-class or localization records in this run.
    assert _records(details, DetectionOutcome.WRONG_CLASS) == []
    assert _records(details, DetectionOutcome.LOCALIZATION) == []


def test_wrong_class_at_high_threshold():
    gts, dets = _build_boxes()
    results = get_pascalvoc_metrics(gts, dets, iou_threshold=0.9, generate_detail=True)
    details = results['details']

    # At .9 the same-class overlap (.75) is below the threshold, but the detection
    # overlaps a ground truth of another class with IoU 1 -> wrong class.
    wrong_records = _records(details, DetectionOutcome.WRONG_CLASS)
    assert len(wrong_records) == 1
    record = wrong_records[0]
    assert record['detection'].get_confidence() == 0.8
    assert record['wrong_class_gt'] is gts[2]
    assert isclose(record['iou'], 1.0)
    assert _records(details, DetectionOutcome.DUPLICATE) == []
    # The .9 detection still matches its ground truth at .9.
    tp_records = _records(details, DetectionOutcome.TP)
    assert len(tp_records) == 1
    assert tp_records[0]['matched_gt'] is gts[0]


def test_localization_when_no_qualifying_overlap():
    # A detection whose best same-class IoU is below the threshold and that does
    # not overlap any other class enough is reported as poor localization.
    gts = [_gt('A', (0, 0, 99, 99))]
    dets = [_det('A', 0.7, (200, 200, 299, 299))]
    results = get_pascalvoc_metrics(gts, dets, iou_threshold=0.5, generate_detail=True)
    loc_records = results['details'][DetectionOutcome.LOCALIZATION]
    assert len(loc_records) == 1
    record = loc_records[0]
    assert record['detection'] is dets[0]
    assert record['matched_gt'] is gts[0]
    assert isclose(record['iou'], 0.0)


def test_ignored_detections_are_listed_separately():
    gts, dets = _build_boxes()
    ignored_det = _det('A', 0.6, (0, 0, 99, 99), is_ignore=True)
    dets = dets + [ignored_det]

    results = get_pascalvoc_metrics(gts, dets, iou_threshold=0.5, generate_detail=True)
    details = results['details']

    ignored_records = details[DetectionOutcome.IGNORED]
    assert len(ignored_records) == 1
    assert ignored_records[0]['detection'] is ignored_det
    # The ignored detection does not show up in any other group.
    accounted = []
    for outcome in DetectionOutcome:
        if outcome == DetectionOutcome.IGNORED:
            continue
        accounted.extend(r['detection'] for r in details[outcome])
    assert all(det is not ignored_det for det in accounted)

    # It must not count as TP or FP: only the two non-ignored detections are scored.
    for class_id, class_result in results['per_class'].items():
        assert float(class_result['total TP']) + float(class_result['total FP']) <= 2


def test_ap_identical_with_and_without_details_across_runs():
    gts, dets = _build_boxes()
    for iou_threshold in (0.5, 0.9):
        plain = get_pascalvoc_metrics(gts,
                                      dets,
                                      iou_threshold=iou_threshold,
                                      method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION)
        detailed = get_pascalvoc_metrics(gts,
                                         dets,
                                         iou_threshold=iou_threshold,
                                         method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
                                         generate_detail=True)
        assert isclose(plain['mAP'], detailed['mAP'])
        for class_id in plain['per_class']:
            assert isclose(plain['per_class'][class_id]['AP'],
                           detailed['per_class'][class_id]['AP'])
            assert isclose(float(plain['per_class'][class_id]['total TP']),
                           float(detailed['per_class'][class_id]['total TP']))
            assert isclose(float(plain['per_class'][class_id]['total FP']),
                           float(detailed['per_class'][class_id]['total FP']))


def test_legacy_output_shape_and_sorting_unchanged_without_details():
    gts, dets = _build_boxes()
    results = get_pascalvoc_metrics(gts, dets, iou_threshold=0.5)
    # No details key is added when the details are not requested.
    assert 'details' not in results
    assert set(results.keys()) == {'per_class', 'mAP'}
    # The detections are still processed in decreasing confidence order.
    class_a = results['per_class']['A']
    # recall must be non-decreasing, reflecting the confidence-sorted processing.
    recall = list(class_a['recall'])
    assert recall == sorted(recall)


def test_ignored_detection_does_not_change_ap_against_legacy():
    gts, dets = _build_boxes()
    ignored_det = _det('A', 0.6, (0, 0, 99, 99), is_ignore=True)
    dets_with_ignore = dets + [ignored_det]

    detailed = get_pascalvoc_metrics(gts,
                                     dets_with_ignore,
                                     iou_threshold=0.5,
                                     generate_detail=True)
    # AP computed over the non-ignored detections only.
    reference = get_pascalvoc_metrics(gts, dets, iou_threshold=0.5)
    assert isclose(detailed['mAP'], reference['mAP'])
    for class_id in reference['per_class']:
        assert isclose(detailed['per_class'][class_id]['AP'],
                       reference['per_class'][class_id]['AP'])
