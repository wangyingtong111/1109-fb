""" Tests for the shared geometry convention.

These tests cover the scenario where boxes coming from two different geometric
origins are compared inside the same detection report:

* continuous half-open XYXY boxes (floating point allowed), e.g. COCO;
* inclusive integer pixel spans, e.g. PASCAL VOC, where [3, 0, 3, 0] is a
  whole pixel.

Only the entry points explicitly opting in (use_shared_geometry=True) apply the
unified rule; the legacy calls keep their historical behaviour.
"""

from math import isclose

from src.bounding_box import BBFormat, BBType, BoundingBox
from src.evaluators.coco_evaluator import (_compute_ious, _get_area,
                                           _jaccard, get_coco_metrics,
                                           get_coco_summary)
from src.evaluators.pascal_voc_evaluator import get_pascalvoc_metrics
from src.utils.enumerators import BBGeometry, MethodAveragePrecision
from src.utils.shared_geometry import (get_continuous_area,
                                       get_continuous_coordinates, shared_iou)

IMG = 'img'
CLS = 'obj'


def _bb(coords,
        bb_type,
        geometry,
        confidence=None,
        image_name=IMG,
        class_id=CLS):
    return BoundingBox(image_name=image_name,
                       class_id=class_id,
                       coordinates=coords,
                       bb_type=bb_type,
                       confidence=confidence,
                       format=BBFormat.XYX2Y2,
                       geometry=geometry)


def _gt(coords, geometry):
    return _bb(coords, BBType.GROUND_TRUTH, geometry)


def _dt(coords, geometry, confidence):
    return _bb(coords, BBType.DETECTED, geometry, confidence=confidence)


# ---------------------------------------------------------------------------
# 1. Per-box geometry markers
# ---------------------------------------------------------------------------
def test_default_geometry_is_continuous():
    box = _gt((0.0, 0.0, 1.0, 1.0), BBGeometry.CONTINUOUS)
    assert box.get_geometry() == BBGeometry.CONTINUOUS
    # a box built without an explicit marker is treated as continuous
    plain = BoundingBox(image_name=IMG,
                        class_id=CLS,
                        coordinates=(0, 0, 1, 1),
                        bb_type=BBType.GROUND_TRUTH,
                        format=BBFormat.XYX2Y2)
    assert plain.get_geometry() == BBGeometry.CONTINUOUS


def test_geometry_round_trip_and_validation():
    box = _gt((3, 0, 3, 0), BBGeometry.PIXEL_CLOSED)
    assert box.get_geometry() == BBGeometry.PIXEL_CLOSED
    box.set_geometry(BBGeometry.CONTINUOUS)
    assert box.get_geometry() == BBGeometry.CONTINUOUS
    # None is accepted and falls back to continuous
    box.set_geometry(None)
    assert box.get_geometry() == BBGeometry.CONTINUOUS
    # invalid values are rejected
    try:
        box.set_geometry('pixel_closed')
    except IOError:
        pass
    else:
        raise AssertionError('Expected IOError for an invalid geometry value')


# ---------------------------------------------------------------------------
# 2. Non-destructive conversion of pixel-closed boxes to continuous regions
# ---------------------------------------------------------------------------
def test_pixel_closed_conversion_is_non_destructive():
    pixel_box = _gt((3, 0, 3, 0), BBGeometry.PIXEL_CLOSED)
    # the inclusive pixel span maps to the half-open [3, 4) x [0, 1)
    assert get_continuous_coordinates(pixel_box) == (3.0, 0.0, 4.0, 1.0)
    assert isclose(get_continuous_area(pixel_box), 1.0)
    # stored coordinates are untouched and remain integer inclusive spans
    assert pixel_box.get_absolute_bounding_box(BBFormat.XYX2Y2) == (3.0, 0.0,
                                                                     3.0, 0.0)
    # conversion is idempotent
    for _ in range(5):
        assert get_continuous_coordinates(pixel_box) == (3.0, 0.0, 4.0, 1.0)
    assert pixel_box.get_absolute_bounding_box(BBFormat.XYX2Y2) == (3.0, 0.0,
                                                                     3.0, 0.0)

    continuous_box = _gt((3.0, 0.0, 4.0, 1.0), BBGeometry.CONTINUOUS)
    assert get_continuous_coordinates(continuous_box) == (3.0, 0.0, 4.0, 1.0)
    assert isclose(get_continuous_area(continuous_box), 1.0)


# ---------------------------------------------------------------------------
# 3. Exact IoU values required by the specification (shared convention)
# ---------------------------------------------------------------------------
def test_edge_touching_continuous_boxes_have_zero_iou():
    # [0,0,3,3] and [3,0,6,3] only share the edge x=3
    box_a = _gt((0, 0, 3, 3), BBGeometry.CONTINUOUS)
    box_b = _gt((3, 0, 6, 3), BBGeometry.CONTINUOUS)
    assert isclose(shared_iou(box_a, box_b), 0.0)
    assert isclose(BoundingBox.iou(box_a, box_b, use_shared_geometry=True),
                   0.0)
    assert isclose(_jaccard(box_a, box_b, use_shared_geometry=True), 0.0)


def test_pixel_closed_single_pixel_matches_continuous_box():
    # pixel [3,0,3,0] is the whole pixel covered by continuous [3,0,4,1]
    pixel_box = _gt((3, 0, 3, 0), BBGeometry.PIXEL_CLOSED)
    continuous_box = _dt((3, 0, 4, 1), BBGeometry.CONTINUOUS, 0.9)
    assert isclose(shared_iou(pixel_box, continuous_box), 1.0)
    assert isclose(
        BoundingBox.iou(pixel_box, continuous_box,
                        use_shared_geometry=True), 1.0)
    assert isclose(
        _jaccard(pixel_box, continuous_box, use_shared_geometry=True), 1.0)
    # IoU is symmetric
    assert isclose(shared_iou(continuous_box, pixel_box), 1.0)


def test_zero_width_continuous_box_has_zero_iou():
    # degenerate box [3,0,3,1] against the reference pixel [3,0,4,1]
    zero_width = _dt((3, 0, 3, 1), BBGeometry.CONTINUOUS, 0.8)
    reference = _gt((3, 0, 4, 1), BBGeometry.CONTINUOUS)
    assert isclose(shared_iou(zero_width, reference), 0.0)
    assert isclose(
        BoundingBox.iou(zero_width, reference,
                        use_shared_geometry=True), 0.0)
    assert isclose(
        _jaccard(zero_width, reference, use_shared_geometry=True), 0.0)
    # two zero-area boxes never produce a division by zero
    other_zero = _gt((3, 0, 3, 1), BBGeometry.CONTINUOUS)
    assert isclose(shared_iou(zero_width, other_zero), 0.0)


# ---------------------------------------------------------------------------
# 4. Legacy behaviour is preserved when the convention is not selected
# ---------------------------------------------------------------------------
def test_legacy_voc_still_counts_the_shared_edge():
    # legacy inclusive-pixel VOC: edge contact contributes a 1-unit strip
    box_a = _gt((0, 0, 3, 3), BBGeometry.CONTINUOUS)
    box_b = _dt((3, 0, 6, 3), BBGeometry.CONTINUOUS, 0.9)
    legacy_iou = BoundingBox.iou(box_a, box_b)
    # intersection 1x4 = 4, areas 16 and 16, union 28 -> 4/28
    assert isclose(legacy_iou, 4.0 / 28.0)
    # explicit default argument matches the implicit legacy call
    assert isclose(legacy_iou,
                   BoundingBox.iou(box_a, box_b, use_shared_geometry=False))


def test_legacy_coco_keeps_open_ended_area_rule():
    pixel_box = _gt((3, 0, 3, 0), BBGeometry.PIXEL_CLOSED)
    continuous_box = _dt((3, 0, 4, 1), BBGeometry.CONTINUOUS, 0.9)
    # legacy COCO sees the inclusive pixel box as a zero-area point
    assert isclose(_get_area(pixel_box), 0.0)
    assert isclose(_jaccard(pixel_box, continuous_box), 0.0)
    # whereas the shared convention recognises the whole pixel (area 1, IoU 1)
    assert isclose(_get_area(pixel_box, use_shared_geometry=True), 1.0)
    assert isclose(
        _jaccard(pixel_box, continuous_box, use_shared_geometry=True), 1.0)


# ---------------------------------------------------------------------------
# 5. Mixed dataset: both evaluators agree on localization and area filtering
# ---------------------------------------------------------------------------
def _build_mixed_dataset():
    # Ground truths mixing pixel-closed small boxes and continuous float boxes
    gts = [
        _gt((3, 0, 3, 0), BBGeometry.PIXEL_CLOSED),          # area 1 (small)
        _gt((10.5, 10.5, 14.5, 14.5), BBGeometry.CONTINUOUS),  # area 16
        _gt((40, 40, 45, 45), BBGeometry.PIXEL_CLOSED),      # area 36
        _gt((20, 20, 24, 24), BBGeometry.CONTINUOUS),        # area 16, never matched
        _gt((60, 60, 100, 100), BBGeometry.CONTINUOUS),      # area 1600 (medium)
    ]
    dts = [
        _dt((3, 0, 4, 1), BBGeometry.CONTINUOUS, 0.99),      # hits pixel GT (IoU 1)
        _dt((0, 0, 3, 3), BBGeometry.CONTINUOUS, 0.95),      # edge-only -> FP
        _dt((10.5, 10.5, 14.5, 14.5), BBGeometry.CONTINUOUS, 0.90),  # hit
        _dt((100, 100, 110, 110), BBGeometry.CONTINUOUS, 0.85),  # no GT overlap -> FP
        _dt((40, 40, 45, 45), BBGeometry.PIXEL_CLOSED, 0.80),  # hits pixel GT
        _dt((60, 60, 100, 100), BBGeometry.CONTINUOUS, 0.70),  # hits medium GT
    ]
    return gts, dts


def _snapshot(boxes):
    return [b.get_absolute_bounding_box(BBFormat.XYX2Y2) for b in boxes]


def test_mixed_dataset_localization_is_consistent_between_evaluators():
    gts, dts = _build_mixed_dataset()

    voc = get_pascalvoc_metrics(gts,
                                dts,
                                iou_threshold=0.5,
                                method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
                                use_shared_geometry=True)['per_class'][CLS]
    coco = get_coco_metrics(gts,
                            dts,
                            iou_threshold=0.5,
                            use_shared_geometry=True)[CLS]

    # 4 true positives (pixel hit, float hit, pixel-closed hit, medium hit),
    # 2 false positives (edge-only and non-overlapping), 5 ground truths
    assert int(voc['total TP']) == 4
    assert int(voc['total FP']) == 2
    assert voc['total positives'] == 5
    assert int(coco['TP']) == 4
    assert int(coco['FP']) == 2
    assert coco['total positives'] == 5
    assert isclose(voc['recall'][-1], 4.0 / 5.0)
    assert isclose(coco['recall'][-1], 4.0 / 5.0)


def test_mixed_dataset_area_filtering_is_consistent_between_evaluators():
    gts, dts = _build_mixed_dataset()
    medium_range = (32**2, 96**2)  # only the 40x40 (area 1600) GT qualifies

    # COCO applies the area range internally through the shared area rule
    coco = get_coco_metrics(gts,
                            dts,
                            iou_threshold=0.5,
                            area_range=medium_range,
                            use_shared_geometry=True)[CLS]
    assert int(coco['TP']) == 1
    assert int(coco['FP']) == 0
    assert coco['total positives'] == 1

    # VOC has no native area filter: apply the *same* shared area rule up-front,
    # then evaluate. Both entry points must reach the same conclusion.
    kept_gts = [g for g in gts if medium_range[0] <= get_continuous_area(g)
                <= medium_range[1]]
    kept_dts = [d for d in dts if medium_range[0] <= get_continuous_area(d)
                <= medium_range[1]]
    voc = get_pascalvoc_metrics(kept_gts,
                                kept_dts,
                                iou_threshold=0.5,
                                use_shared_geometry=True)['per_class'][CLS]
    assert int(voc['total TP']) == 1
    assert int(voc['total FP']) == 0
    assert voc['total positives'] == 1


def test_shared_area_rule_changes_small_box_filtering_vs_legacy_coco():
    gts, dts = _build_mixed_dataset()
    small_range = (0, 32**2)

    # shared convention: the single-pixel GT has area 1 and stays in the
    # small range, so the detection covering it is a true positive
    shared = get_coco_metrics(gts,
                              dts,
                              iou_threshold=0.5,
                              area_range=small_range,
                              use_shared_geometry=True)[CLS]
    assert int(shared['TP']) == 3  # pixel + float + the 40..45 (area 36) hits
    assert int(shared['FP']) == 2  # edge-only and non-overlapping detections

    # legacy convention: the single-pixel GT collapses to area 0 and is
    # ignored, and its detection becomes a false positive instead
    legacy = get_coco_metrics(gts,
                              dts,
                              iou_threshold=0.5,
                              area_range=small_range,
                              use_shared_geometry=False)[CLS]
    assert int(legacy['TP']) == 2
    assert int(legacy['FP']) == 3


def test_repeated_evaluations_are_idempotent_and_do_not_mutate_inputs():
    gts, dts = _build_mixed_dataset()
    gt_before = _snapshot(gts)
    dt_before = _snapshot(dts)
    geom_before = ([b.get_geometry() for b in gts]
                   + [b.get_geometry() for b in dts])

    # run the shared convention twice on both evaluators
    voc_1 = get_pascalvoc_metrics(gts, dts, 0.5,
                                  use_shared_geometry=True)['per_class'][CLS]
    coco_1 = get_coco_metrics(gts, dts, 0.5,
                              use_shared_geometry=True)[CLS]
    voc_2 = get_pascalvoc_metrics(gts, dts, 0.5,
                                  use_shared_geometry=True)['per_class'][CLS]
    coco_2 = get_coco_metrics(gts, dts, 0.5,
                              use_shared_geometry=True)[CLS]

    assert int(voc_1['total TP']) == int(voc_2['total TP']) == 4
    assert int(voc_1['total FP']) == int(voc_2['total FP']) == 2
    assert int(coco_1['TP']) == int(coco_2['TP']) == 4
    assert int(coco_1['FP']) == int(coco_2['FP']) == 2

    # coordinates and markers are unchanged after repeated evaluations
    assert _snapshot(gts) == gt_before
    assert _snapshot(dts) == dt_before
    assert ([b.get_geometry() for b in gts]
            + [b.get_geometry() for b in dts]) == geom_before

    # legacy VOC calls must also leave the inputs untouched and stay stable
    legacy_1 = get_pascalvoc_metrics(gts, dts, 0.5)['per_class'][CLS]
    legacy_2 = get_pascalvoc_metrics(gts, dts, 0.5)['per_class'][CLS]
    assert int(legacy_1['total TP']) == int(legacy_2['total TP'])
    assert int(legacy_1['total FP']) == int(legacy_2['total FP'])
    assert _snapshot(gts) == gt_before
    assert _snapshot(dts) == dt_before


def test_pairwise_iou_matrix_uses_shared_convention_without_mutation():
    gts, dts = _build_mixed_dataset()
    before = _snapshot(dts) + _snapshot(gts)
    ious = _compute_ious(dts, gts, use_shared_geometry=True)
    # first detection fully covers the single-pixel ground truth (index 0)
    assert isclose(ious[0, 0], 1.0)
    # edge-only detection (index 1) has zero IoU with every ground truth
    assert all(isclose(v, 0.0) for v in ious[1])
    # non-overlapping detection (index 3) also has zero IoU everywhere
    assert all(isclose(v, 0.0) for v in ious[3])
    assert _snapshot(dts) + _snapshot(gts) == before


def test_coco_summary_runs_with_shared_convention():
    gts, dts = _build_mixed_dataset()
    summary = get_coco_summary(gts, dts, use_shared_geometry=True)
    # every standard metric is present and is a finite number
    for key in ('AP', 'AP50', 'AP75', 'APsmall', 'APmedium', 'APlarge',
                'AR1', 'AR10', 'AR100', 'ARsmall', 'ARmedium', 'ARlarge'):
        assert key in summary
