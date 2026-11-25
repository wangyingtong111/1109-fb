""" Tests for sub-pixel normalized boxes, clipped views and the legacy pixel path.

A detection model usually outputs normalized (center x, center y, width,
height) values. These tests cover the requirement of keeping those floats all
the way into the matching/scoring step, of exposing a clipped (visible) view of
a box without ever writing it back to the unclipped normalized value, and of
keeping the legacy BoundingBox pixel calls quantized.

Three scenarios are covered:

1. normal     - a sub-pixel box round-trips absolute -> relative without losing
                precision (100x50 image, center x=.5, width .005 -> borders
                49.75 and 50.25).
2. conflict   - the same box is read as a clipped view for scoring and as an
                unclipped relative value afterwards, in both orders: the clipped
                width is .005 with IoU 1, and the unclipped width stays .01.
3. regression - legacy pixel calls keep the historical whole-pixel rounding.
"""

from math import isclose

from src.bounding_box import BoundingBox
from src.evaluators.coco_evaluator import (_compute_ious, _get_area, _jaccard,
                                           get_coco_metrics, get_coco_summary)
from src.evaluators.pascal_voc_evaluator import get_pascalvoc_metrics
from src.utils.enumerators import (BBFormat, BBType, CoordinatesPrecision,
                                   CoordinatesType)

IMG = "img"
CLS = "obj"
SIZE = (100, 50)  # width x height


def _relative(cx, cy, w, h, detected=False, confidence=None, precision=None):
    kwargs = {}
    if precision is not None:
        kwargs["precision"] = precision
    return BoundingBox(IMG,
                       CLS,
                       (cx, cy, w, h),
                       type_coordinates=CoordinatesType.RELATIVE,
                       img_size=SIZE,
                       bb_type=BBType.DETECTED if detected else BBType.GROUND_TRUTH,
                       confidence=confidence,
                       **kwargs)


def _abs_xyxy(box):
    return box.get_absolute_bounding_box(BBFormat.XYX2Y2)


def _snapshot(*boxes):
    return [(b.get_absolute_bounding_box(BBFormat.XYX2Y2),
             b.get_absolute_bounding_box(BBFormat.XYWH),
             b.get_relative_bounding_box(),
             b.get_precision()) for b in boxes]


# ---------------------------------------------------------------------------
# 1. Normal: sub-pixel borders and lossless relative round-trip
# ---------------------------------------------------------------------------
def test_exact_relative_box_keeps_subpixel_borders():
    box = _relative(.5, .5, .005, .01, precision=CoordinatesPrecision.EXACT)
    x1, y1, x2, y2 = _abs_xyxy(box)
    # 100 * .5 = 50 is the center; .005 * 100 = .5 is the width
    assert x1 == 49.75
    assert x2 == 50.25
    assert isclose(x2 - x1, 0.5)
    assert box.get_absolute_bounding_box(BBFormat.XYWH)[2] == 0.5


def test_exact_relative_width_round_trips():
    box = _relative(.5, .5, .005, .01, precision=CoordinatesPrecision.EXACT)
    cx, cy, w, h = box.get_relative_bounding_box()
    assert w == .005
    assert cx == .5
    assert h == .01
    assert box.get_relative_bounding_box() == (cx, cy, w, h)
    # an explicit image size equal to the stored one must not quantize either
    assert box.get_relative_bounding_box(SIZE)[2] == .005


def test_exact_precision_is_not_applied_to_absolute_boxes():
    box = BoundingBox(IMG, CLS, (49.75, 24.75, 50.25, 25.25),
                      format=BBFormat.XYX2Y2,
                      type_coordinates=CoordinatesType.ABSOLUTE,
                      img_size=SIZE,
                      precision=CoordinatesPrecision.EXACT)
    assert _abs_xyxy(box) == (49.75, 24.75, 50.25, 25.25)


def test_precision_round_trip_and_validation():
    box = _relative(.5, .5, .005, .01)
    assert box.get_precision() == CoordinatesPrecision.QUANTIZED
    box.set_precision(CoordinatesPrecision.EXACT)
    assert box.get_precision() == CoordinatesPrecision.EXACT
    box.set_precision(None)
    assert box.get_precision() == CoordinatesPrecision.QUANTIZED
    try:
        box.set_precision("exact")
    except IOError:
        pass
    else:
        raise AssertionError("Expected IOError for an invalid precision value")


def test_exact_iou_uses_subpixel_geometry():
    # two boxes only .25 pixel apart in width are distinguishable
    gt = _relative(.5, .5, .005, .01, precision=CoordinatesPrecision.EXACT)
    det = _relative(.5, .5, .0025, .01, detected=True, confidence=.9,
                    precision=CoordinatesPrecision.EXACT)
    iou = BoundingBox.iou(gt, det, continuous=True)
    assert isclose(iou, 0.5)


# ---------------------------------------------------------------------------
# 2. Conflict: clipped view for scoring vs unclipped relative value
# ---------------------------------------------------------------------------
def _left_edge_box():
    """ Center x=0 with relative width .01: borders -0.5 .. 0.5 on a 100px image. """
    return _relative(0.0, .5, .01, .01, precision=CoordinatesPrecision.EXACT)


def _equal_visible_box():
    """ The visible part of the left-edge box: relative width .005. """
    return _relative(.0025, .5, .005, .01, precision=CoordinatesPrecision.EXACT)


def test_clipped_view_has_half_width_and_keeps_unclipped_width():
    box = _left_edge_box()
    assert box.get_relative_bounding_box()[2] == .01
    assert box.get_relative_bounding_box(clip=True)[2] == .005
    assert _abs_xyxy(box) == (-0.5, 24.75, 0.5, 25.25)
    assert box.get_clipped_bounding_box().get_absolute_bounding_box(
        BBFormat.XYX2Y2) == (0.0, 24.75, 0.5, 25.25)


def test_clip_scoring_then_read_unclipped_width():
    box = _left_edge_box()
    visible = _equal_visible_box()
    iou = BoundingBox.iou(box, visible, continuous=True, clip=True)
    assert isclose(iou, 1.0)
    # reading the unclipped value afterwards is unaffected by the clipped scoring
    assert box.get_relative_bounding_box()[2] == .01
    assert box.get_relative_bounding_box(clip=True)[2] == .005


def test_read_unclipped_width_then_clip_scoring():
    box = _left_edge_box()
    visible = _equal_visible_box()
    assert box.get_relative_bounding_box()[2] == .01
    iou = BoundingBox.iou(box, visible, continuous=True, clip=True)
    assert isclose(iou, 1.0)
    assert box.get_relative_bounding_box()[2] == .01


def test_clip_scoring_matches_a_plain_visible_box():
    # scoring the clipped view must equal scoring the equivalent visible box
    box = _left_edge_box()
    visible = _equal_visible_box()
    direct = BoundingBox.iou(visible, visible, continuous=True)
    clipped = BoundingBox.iou(box, visible, continuous=True, clip=True)
    assert isclose(direct, 1.0)
    assert isclose(clipped, direct)


def test_voc_metrics_are_order_independent_and_non_destructive():
    for _ in range(2):  # both reading orders
        gt = _left_edge_box()
        det = _relative(.0025, .5, .005, .01, detected=True, confidence=.9,
                        precision=CoordinatesPrecision.EXACT)
        before = _snapshot(gt, det)
        # the detection is the equal visible box; match on the clipped view
        res = get_pascalvoc_metrics([gt], [det],
                                    iou_threshold=0.5,
                                    continuous=True,
                                    clip=True)["per_class"][CLS]
        assert int(res["total TP"]) == 1
        assert int(res["total FP"]) == 0
        assert _snapshot(gt, det) == before
        assert gt.get_relative_bounding_box()[2] == .01
        assert gt.get_relative_bounding_box(clip=True)[2] == .005


def test_coco_metrics_are_order_independent_and_non_destructive():
    for _ in range(2):
        gt = _left_edge_box()
        det = _relative(.0025, .5, .005, .01, detected=True, confidence=.9,
                        precision=CoordinatesPrecision.EXACT)
        before = _snapshot(gt, det)
        res = get_coco_metrics([gt], [det],
                               iou_threshold=0.5,
                               continuous=True,
                               clip=True)[CLS]
        assert int(res["TP"]) == 1
        assert int(res["FP"]) == 0
        assert _snapshot(gt, det) == before
        assert gt.get_relative_bounding_box()[2] == .01


def test_repeated_shared_calls_stay_stable():
    gt = _left_edge_box()
    det = _relative(.0025, .5, .005, .01, detected=True, confidence=.9,
                    precision=CoordinatesPrecision.EXACT)
    before = _snapshot(gt, det)
    first = get_coco_summary([gt], [det], continuous=True, clip=True)
    second = get_coco_summary([gt], [det], continuous=True, clip=True)
    assert isclose(first["AP50"], second["AP50"])
    assert isclose(first["AP50"], 1.0)
    assert _snapshot(gt, det) == before


def test_unclipped_geometry_is_never_overwritten_by_a_clipped_view():
    box = _left_edge_box()
    unclipped = box.get_absolute_bounding_box(BBFormat.XYX2Y2)
    # build several clipped views, in different shapes
    _ = box.get_relative_bounding_box(clip=True)
    _ = box.get_absolute_bounding_box(BBFormat.XYX2Y2, clip=True)
    _ = box.get_clipped_bounding_box()
    _ = box.get_clipped_bounding_box((200, 100))
    assert box.get_absolute_bounding_box(BBFormat.XYX2Y2) == unclipped
    assert box.get_relative_bounding_box() == (0.0, .5, .01, .01)


# ---------------------------------------------------------------------------
# 3. Regression: legacy pixel calls keep quantizing
# ---------------------------------------------------------------------------
def test_legacy_small_box_still_uses_integer_corners():
    box = _relative(.5, .5, .005, .01)
    assert box.get_precision() == CoordinatesPrecision.QUANTIZED
    # legacy pixel conversion rounds the sub-pixel borders away
    assert _abs_xyxy(box) == (50.0, 25.0, 50.0, 25.0)
    assert box.get_absolute_bounding_box(BBFormat.XYWH) == (50.0, 25.0, 0.0, 0.0)


def test_legacy_pixel_iou_and_area_are_unchanged():
    # bounding boxes on whole pixels behave exactly as before
    a = BoundingBox(IMG, CLS, (0, 0, 3, 3), format=BBFormat.XYX2Y2,
                    type_coordinates=CoordinatesType.ABSOLUTE)
    b = BoundingBox(IMG, CLS, (3, 0, 6, 3), format=BBFormat.XYX2Y2,
                    type_coordinates=CoordinatesType.ABSOLUTE)
    # legacy VOC counts the shared edge as a 1-unit strip
    assert isclose(BoundingBox.iou(a, b), 4.0 / 28.0)
    # legacy inclusive area
    assert a.get_area() == 16
    # legacy COCO open-ended area of a point-like box is zero
    point = BoundingBox(IMG, CLS, (3, 0, 3, 0), format=BBFormat.XYX2Y2,
                        type_coordinates=CoordinatesType.ABSOLUTE)
    assert _get_area(point) == 0


def test_legacy_coco_and_voc_still_agree_on_whole_pixel_boxes():
    gt = BoundingBox(IMG, CLS, (0, 0, 16, 16), format=BBFormat.XYX2Y2,
                     type_coordinates=CoordinatesType.ABSOLUTE)
    det = BoundingBox(IMG, CLS, (0, 0, 16, 16), format=BBFormat.XYX2Y2,
                      type_coordinates=CoordinatesType.ABSOLUTE,
                      bb_type=BBType.DETECTED, confidence=.9)
    assert BoundingBox.iou(gt, det) == 1.0
    assert get_coco_summary([gt], [det])["AP50"] == 1.0
    assert get_pascalvoc_metrics([gt], [det])["per_class"][CLS]["AP"] == 1.0


def test_continuous_and_legacy_paths_can_coexist_without_mutation():
    gt = _relative(0.0, .5, .01, .01, precision=CoordinatesPrecision.EXACT)
    det = _relative(.0025, .5, .005, .01, detected=True, confidence=.9,
                    precision=CoordinatesPrecision.EXACT)
    before = _snapshot(gt, det)
    legacy_iou = BoundingBox.iou(gt, det)
    shared_iou = BoundingBox.iou(gt, det, continuous=True)
    assert _snapshot(gt, det) == before
    # the unclipped, edge-crossing box is not a perfect match under the legacy
    # inclusive rule, but is under the continuous convention once clipped
    assert legacy_iou < 1.0
    assert isclose(BoundingBox.iou(gt, det, continuous=True, clip=True), 1.0)
    assert shared_iou >= 0.0


def test_pairwise_iou_matrix_uses_shared_convention_without_mutation():
    gt = _left_edge_box()
    det = _relative(.0025, .5, .005, .01, detected=True, confidence=.9,
                    precision=CoordinatesPrecision.EXACT)
    before = _snapshot(gt, det)
    ious = _compute_ious([det], [gt], continuous=True, clip=True)
    assert isclose(ious[0, 0], 1.0)
    assert _snapshot(gt, det) == before
    # and the unclipped pairwise jaccard does not see a perfect match
    assert _jaccard(det, gt, continuous=True) < 1.0


def test_converter_accepts_exact_precision_for_relative_boxes(tmp_path):
    import src.utils.converter as converter

    images_dir = "toyexample/images"
    annotation_file = tmp_path / "2007_000733.txt"
    # model output: class_id confidence cx cy w h
    annotation_file.write_text("0 0.9 0.5 0.5 0.005 0.01\n")
    boxes = converter.text2bb(str(tmp_path),
                              bb_type=BBType.DETECTED,
                              bb_format=BBFormat.YOLO,
                              type_coordinates=CoordinatesType.RELATIVE,
                              img_dir=images_dir,
                              precision=CoordinatesPrecision.EXACT)
    assert len(boxes) == 1
    box = boxes[0]
    # 450x435 image: .5 * 450 = 225 is the center, .005 * 450 = 2.25 the width
    assert isclose(box.get_absolute_bounding_box(BBFormat.XYX2Y2)[0], 223.875)
    assert isclose(box.get_absolute_bounding_box(BBFormat.XYX2Y2)[2], 226.125)
    assert box.get_relative_bounding_box()[2] == .005
    # the legacy default still quantizes
    legacy = converter.text2bb(str(tmp_path),
                               bb_type=BBType.DETECTED,
                               bb_format=BBFormat.YOLO,
                               type_coordinates=CoordinatesType.RELATIVE,
                               img_dir=images_dir)
    assert legacy[0].get_absolute_bounding_box(BBFormat.XYX2Y2) == (224.0, 215.0,
                                                                   226.0, 220.0)
    # the legacy conversion also drifts in relative terms, the exact one does not
    assert legacy[0].get_relative_bounding_box()[2] != .005


def test_exact_converter_keeps_boxes_that_overflow_the_frame():
    import src.utils.converter as converter

    images_dir = "toyexample/images"
    # A model output whose relative height is slightly larger than 1.0: the box
    # overflows the bottom border. It is a valid detection, so it must not be
    # silently dropped just because one absolute border is out of the frame.
    relative_dir = "toyexample/dets_classname_rel_xcycwh"
    exact = converter.text2bb(relative_dir,
                              bb_type=BBType.DETECTED,
                              bb_format=BBFormat.YOLO,
                              type_coordinates=CoordinatesType.RELATIVE,
                              img_dir=images_dir,
                              precision=CoordinatesPrecision.EXACT)
    legacy = converter.text2bb(relative_dir,
                               bb_type=BBType.DETECTED,
                               bb_format=BBFormat.YOLO,
                               type_coordinates=CoordinatesType.RELATIVE,
                               img_dir=images_dir)
    assert len(exact) == len(legacy)
    assert [b.get_image_name() for b in exact] == [b.get_image_name() for b in legacy]
    # the overflowing box is present and its clipped view is inside the image
    overflowing = [b for b in exact if b.get_relative_bounding_box()[3] > 1.0]
    assert len(overflowing) == 1
    clipped = overflowing[0].get_clipped_bounding_box()
    width, height = overflowing[0].get_image_size()
    x1, y1, x2, y2 = clipped.get_absolute_bounding_box(BBFormat.XYX2Y2)
    assert 0.0 <= x1 <= x2 <= width
    assert 0.0 <= y1 <= y2 <= height

# ---------------------------------------------------------------------------
# 4. Clipped-view edge cases
# ---------------------------------------------------------------------------
def test_clipping_is_idempotent_and_preserves_image_size():
    box = _left_edge_box()
    clipped_once = box.get_clipped_bounding_box()
    clipped_twice = clipped_once.get_clipped_bounding_box()
    expected = (0.0, 24.75, 0.5, 25.25)
    assert clipped_once.get_absolute_bounding_box(BBFormat.XYX2Y2) == expected
    assert clipped_twice.get_absolute_bounding_box(BBFormat.XYX2Y2) == expected
    assert clipped_once.get_relative_bounding_box() == (.0025, .5, .005, .01)
    assert clipped_twice.get_relative_bounding_box() == (.0025, .5, .005, .01)
    # the original image size, with its original type, is kept
    assert clipped_once.get_image_size() == SIZE


def test_box_fully_outside_the_image_clips_to_zero_area():
    box = _relative(-0.005, .5, .001, .01, precision=CoordinatesPrecision.EXACT)
    assert box.get_absolute_bounding_box(BBFormat.XYX2Y2, clip=True) == (0.0, 24.75,
                                                                        0.0, 25.25)
    assert box.get_area(continuous=True, clip=True) == 0.0
    assert BoundingBox.iou(box, box, continuous=True, clip=True) == 0.0
    # the unclipped geometry of the outside box is untouched
    assert box.get_relative_bounding_box()[2] == .001


def test_clip_only_mode_matches_the_specification_regardless_of_border_rule():
    # the specification asks for IoU 1 on the clipped view; both the legacy and
    # the continuous border rules reach it for this box, but only the
    # continuous one does so without counting the shared edge
    box = _left_edge_box()
    visible = _equal_visible_box()
    assert isclose(BoundingBox.iou(box, visible, clip=True), 1.0)
    assert isclose(BoundingBox.iou(box, visible, continuous=True, clip=True), 1.0)
    # without clipping, the continuous rule does not consider it a perfect match
    assert isclose(BoundingBox.iou(box, visible, continuous=True), 0.5)
