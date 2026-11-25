"""Sharded per-image COCO evaluation: mergeability, tie order, conflicts.

These tests encode the acceptance scenarios for UID 1547:

* a shard whose image has no ground truth still keeps its high-score FP;
* equal-score FP/TP ties follow the original image identity order, so
  reversing shard arrival must not change the curve;
* every IoU threshold / area bracket matches the one-shot evaluator;
* repeated image identities are a conflict (ValueError);
* the ordinary one-shot entry point keeps its fields and behavior.
"""

import numpy as np
import pytest

from src.bounding_box import BBType, BoundingBox
from src.evaluators import coco_evaluator as coco


def absolute(box, image="im", category="x", confidence=None):
    if confidence is None:
        return BoundingBox(image, category, box)
    return BoundingBox(image, category, box,
                       bb_type=BBType.DETECTED, confidence=confidence)


def fixture(scores=(0.99, 0.5), partial=False):
    # Detection on image "a" has no ground truth there.  Image "b" has one
    # ground-truth box and one matching detection.  In the documented input
    # order the "a" detection precedes the "b" detection.
    gt = [absolute((0, 0, 8, 8), "b")]
    hit = (0, 0, 8, 6) if partial else (0, 0, 8, 8)
    dt = [
        absolute((0, 0, 8, 8), "a", confidence=scores[0]),
        absolute(hit, "b", confidence=scores[1]),
    ]
    return gt, dt


def shards_for(gt, dt):
    fp = dt[0]
    hit = dt[1]
    sa = coco.evaluate_coco_shard([], [fp], image_id="a")
    sb = coco.evaluate_coco_shard(gt, [hit], image_id="b")
    return sa, sb


def test_high_score_false_positive_on_empty_image_is_kept():
    gt, dt = fixture((0.99, 0.5))
    sa, sb = shards_for(gt, dt)
    assert sa["scores"] == [pytest.approx(0.99)]
    assert sa["matched"] == [False]
    merged = coco.merge_shards([sa, sb])
    assert merged["AP50"] == pytest.approx(0.5)
    # arrival order does not matter for distinct identities/scores
    assert coco.merge_shards([sb, sa])["AP50"] == pytest.approx(0.5)
    assert merged["AP50"] == pytest.approx(coco.get_coco_summary(gt, dt)["AP50"])


def test_equal_score_ties_follow_original_image_order_not_arrival():
    gt, dt = fixture((0.5, 0.5))
    one_shot = coco.get_coco_summary(gt, dt)
    sa, sb = shards_for(gt, dt)  # a == FP first in original order

    forward = coco.merge_coco_shards([sa, sb])
    reversed_arrival = coco.merge_coco_shards([sb, sa])

    for key in coco.COCO_SUMMARY_KEYS:
        a, b, ref = forward[key], reversed_arrival[key], one_shot[key]
        if np.isnan(ref):
            assert np.isnan(a) and np.isnan(b)
        else:
            assert a == pytest.approx(ref, abs=1e-10)
            assert b == pytest.approx(ref, abs=1e-10)

    filters = [
        (0.5, (0, np.inf)),
        (0.75, (0, np.inf)),
        (0.5, (0, 32 ** 2)),
    ]
    for threshold, area in filters:
        expected = coco.get_coco_metrics(
            gt, dt, iou_threshold=threshold, area_range=area)["x"]
        for arrival in ([sa, sb], [sb, sa]):
            got = coco.merge_coco_shard_config(
                arrival, iou_threshold=threshold, area_range=area)["x"]
            assert got["AP"] == pytest.approx(expected["AP"], abs=1e-12)
            np.testing.assert_allclose(
                got["interpolated precision"],
                expected["interpolated precision"],
                rtol=1e-12, atol=1e-12)


def test_duplicate_image_identity_is_a_value_error_conflict():
    gt, dt = fixture((0.5, 0.5))
    sa, sb = shards_for(gt, dt)
    with pytest.raises(ValueError):
        coco.merge_coco_shards([sa, sb, sa])
    with pytest.raises(coco.DuplicateImageError):
        coco.merge_shards([sa, sa])

    stats = coco.evaluate_image_stats(gt, dt, 0.5, (0, 32 ** 2))
    with pytest.raises(ValueError):
        coco.merge_image_stats(stats, ["a", "b", "a"])


def test_probe_style_stats_merge_matches_reference_curve():
    gt, dt = fixture((0.5, 0.5), partial=True)
    stats = coco.shard_stats(gt, dt, 0.5, (0, 32 ** 2))
    assert coco.merge_image_stats(stats, ["b", "a"])["AP"] == pytest.approx(0.5)
    assert coco.merge_image_stats(stats, ["a", "b"])["AP"] == pytest.approx(0.5)


def test_shard_by_image_reversed_matches_one_shot_random():
    rng = np.random.default_rng(7)
    for seed in range(8):
        gt, dt = [], []
        images = [f"img{i}" for i in range(4)]
        for image in images:
            for _ in range(int(rng.integers(0, 3))):
                x, y = rng.uniform(0, 40, 2)
                gt.append(absolute((x, y, rng.uniform(2, 18), rng.uniform(2, 18)),
                                    image, "c"))
            for _ in range(int(rng.integers(0, 3))):
                x, y = rng.uniform(0, 40, 2)
                dt.append(absolute((x, y, rng.uniform(2, 18), rng.uniform(2, 18)),
                                   image, "c", float(rng.uniform(0.05, 1))))
        summary = coco.get_coco_summary(gt, dt)
        shards = coco.shard_coco_by_image(gt, dt)
        merged = coco.merge_coco_shards(list(reversed(shards)))
        for key in coco.COCO_SUMMARY_KEYS:
            ref, got = float(summary[key]), float(merged[key])
            if np.isnan(ref):
                assert np.isnan(got)
            else:
                assert got == pytest.approx(ref, abs=1e-8), (seed, key)


def test_shard_is_json_mergeable_without_boxes():
    gt, dt = fixture((0.5, 0.5))
    sa, sb = shards_for(gt, dt)
    restored = coco.CocoShard.from_json(sa.to_json())
    merged = coco.merge_coco_shards([restored, sb])
    assert merged["AP50"] == pytest.approx(0.5)


def test_one_shot_exact_match_still_ap50_one():
    gt = [BoundingBox("a", "car", (0, 0, 16, 16))]
    dt = [BoundingBox("a", "car", (0, 0, 16, 16),
                      bb_type=BBType.DETECTED, confidence=0.9)]
    assert BoundingBox.iou(gt[0], dt[0]) == 1
    assert coco.get_coco_summary(gt, dt)["AP50"] == 1
