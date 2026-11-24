"""Tests for the hierarchical (parent/leaf) detection evaluation.

The scenarios mirror the requirements:

* one entity carries two same-geometry leaf annotations ('maple' and 'oak'),
  both mapping to parent 'tree';
* another entity sits at the very same coordinates, so only the first entity's
  'oak' label is hit by a detection placed there;
* the same entity id in another image is an independent instance;
* a second prediction on the same parent entity cannot create a second hit.
"""

from math import isclose

from src.bounding_box import BoundingBox
from src.evaluators.hierarchical_evaluator import (
    get_hierarchical_pascalvoc_metrics, map_bounding_boxes_to_parent)
from src.evaluators.pascal_voc_evaluator import get_pascalvoc_metrics
from src.utils.enumerators import BBFormat, BBType, MethodAveragePrecision

# All boxes share the same geometry; the geometry itself is therefore not what
# distinguishes the entities - the (image, entity_id) pair is.
COORDS = (10.0, 20.0, 30.0, 40.0)
MAP = {'maple': 'tree', 'oak': 'tree'}


def _gt(image, label, entity):
    return BoundingBox(image_name=image,
                       class_id=label,
                       coordinates=COORDS,
                       bb_type=BBType.GROUND_TRUTH,
                       format=BBFormat.XYWH,
                       entity_id=entity)


def _det(image, label, confidence=0.9):
    return BoundingBox(image_name=image,
                       class_id=label,
                       coordinates=COORDS,
                       bb_type=BBType.DETECTED,
                       confidence=confidence,
                       format=BBFormat.XYWH)


def test_sibling_leaf_annotations_merge_into_one_parent_entity():
    # Image A: entity 1 has both a 'maple' and an 'oak' annotation (same
    # geometry); entity 2 happens to lie at the same coordinates.
    gts = [
        _gt('A', 'maple', 1),
        _gt('A', 'oak', 1),
        _gt('A', 'maple', 2),
    ]
    # Only an 'oak' detection at the shared coordinates.
    dets = [_det('A', 'oak')]

    res = get_hierarchical_pascalvoc_metrics(
        gts, dets, MAP, iou_threshold=0.5,
        method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION)

    tree = res['parent']['per_class']['tree']
    # Parent view: two positives (two entities), one hit.
    assert tree['total positives'] == 2
    assert tree['total TP'] == 1
    assert tree['total FP'] == 0

    # Leaf view stays exactly as before: maple 0/2 with the unmatched maple
    # still visible, oak 1/1.
    leaf = res['leaf']['per_class']
    assert leaf['maple']['total positives'] == 2
    assert leaf['maple']['total TP'] == 0
    assert leaf['oak']['total positives'] == 1
    assert leaf['oak']['total TP'] == 1

    # The parent denominator must be reproducible from the entity report,
    # and it differs from the raw amount of mapped ground-truth boxes (3).
    entities = res['parent']['entities']['tree']
    assert entities['total positives'] == 2
    assert len(entities['entities']) == 2
    merged = next(e for e in entities['entities'] if e['entity_id'] == 1)
    assert merged['leaf boxes'] == 2
    assert merged['leaf classes'] == ['maple', 'oak']
    # Leaf denominator is exposed as well.
    assert res['parent']['leaf positives'] == {'maple': 2, 'oak': 1}


def test_second_prediction_on_same_parent_entity_is_a_false_positive():
    gts = [_gt('A', 'maple', 1), _gt('A', 'oak', 1), _gt('A', 'maple', 2)]
    # The first entity now gets both an oak and a maple prediction. The group
    # of predictions on this parent entity can only contribute one hit.
    dets = [_det('A', 'oak', 0.95), _det('A', 'maple', 0.90)]

    res = get_hierarchical_pascalvoc_metrics(gts, dets, MAP, iou_threshold=0.5)
    tree = res['parent']['per_class']['tree']
    assert tree['total positives'] == 2
    assert tree['total TP'] == 1
    assert tree['total FP'] == 1

    # Leaf level keeps the standard per-class matching unchanged: the oak
    # prediction hits the oak box, while the maple prediction matches one of
    # the two coincident maple boxes (the other maple stays a miss and counts
    # in the maple denominator).
    leaf = res['leaf']['per_class']
    assert leaf['oak']['total TP'] == 1
    assert leaf['maple']['total positives'] == 2
    assert leaf['maple']['total TP'] == 1
    assert leaf['maple']['total FP'] == 0


def test_same_entity_id_in_another_image_is_an_independent_instance():
    gts = [
        _gt('A', 'maple', 1),
        _gt('A', 'oak', 1),
        _gt('B', 'maple', 1),  # same id, other image -> independent instance
    ]
    # One detection per image, each matching its local entity 1.
    dets = [_det('A', 'oak', 0.9), _det('B', 'oak', 0.8)]

    res = get_hierarchical_pascalvoc_metrics(gts, dets, MAP, iou_threshold=0.5)
    tree = res['parent']['per_class']['tree']
    assert tree['total positives'] == 2
    assert tree['total TP'] == 2
    assert tree['total FP'] == 0

    images = {e['image'] for e in res['parent']['entities']['tree']['entities']}
    assert images == {'A', 'B'}


def test_leaf_view_identical_to_standard_evaluation_and_inputs_untouched():
    gts = [_gt('A', 'maple', 1), _gt('A', 'oak', 1), _gt('A', 'maple', 2)]
    dets = [_det('A', 'oak', 0.95), _det('A', 'maple', 0.90)]
    original_gt_labels = [bb.get_class_id() for bb in gts]
    original_det_labels = [bb.get_class_id() for bb in dets]

    res = get_hierarchical_pascalvoc_metrics(
        gts, dets, MAP, iou_threshold=0.5,
        method=MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION)
    direct = get_pascalvoc_metrics(
        gts, dets, iou_threshold=0.5,
        method=MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION)

    for c, direct_res in direct['per_class'].items():
        leaf_res = res['leaf']['per_class'][c]
        assert leaf_res['total positives'] == direct_res['total positives']
        assert leaf_res['total TP'] == direct_res['total TP']
        assert leaf_res['total FP'] == direct_res['total FP']
        assert isclose(leaf_res['AP'], direct_res['AP'])
    assert isclose(res['leaf']['mAP'], direct['mAP'])

    # Original boxes and labels are preserved for other evaluations.
    assert [bb.get_class_id() for bb in gts] == original_gt_labels
    assert [bb.get_class_id() for bb in dets] == original_det_labels
    assert all(bb.get_entity_id() is not None for bb in gts)


def test_mapping_returns_independent_clones():
    bb = _gt('A', 'maple', 1)
    mapped = map_bounding_boxes_to_parent([bb], MAP)
    assert len(mapped) == 1
    assert mapped[0].get_class_id() == 'tree'
    # Original is untouched and clone geometry/entity are preserved.
    assert bb.get_class_id() == 'maple'
    assert mapped[0].get_entity_id() == 1
    assert mapped[0].get_absolute_bounding_box(BBFormat.XYWH) == COORDS
    mapped[0].set_class_id('bush')
    assert bb.get_class_id() == 'maple'


def test_coincident_boxes_without_entity_id_stay_separate():
    # No entity id -> every box is its own entity, standard per-box matching.
    gts = [_gt('A', 'maple', None), _gt('A', 'oak', None)]
    dets = [_det('A', 'oak', 0.9)]
    res = get_hierarchical_pascalvoc_metrics(gts, dets, MAP, iou_threshold=0.5)
    tree = res['parent']['per_class']['tree']
    assert tree['total positives'] == 2
    assert tree['total TP'] == 1
    assert tree['total FP'] == 0


def test_parent_view_matches_standard_evaluator_without_duplicate_entities():
    # One box per entity and all leaves mapped to 'tree': parent counts must
    # coincide with running the standard evaluator on 'tree'-labelled boxes.
    gts = [_gt('A', 'maple', 1), _gt('A', 'oak', 2), _gt('B', 'oak', 3)]
    dets = [_det('A', 'oak', 0.9), _det('B', 'maple', 0.8)]

    res = get_hierarchical_pascalvoc_metrics(gts, dets, MAP, iou_threshold=0.5)
    gt_tree = map_bounding_boxes_to_parent(gts, MAP)
    det_tree = map_bounding_boxes_to_parent(dets, MAP)
    direct = get_pascalvoc_metrics(gt_tree, det_tree, iou_threshold=0.5)['per_class']['tree']

    tree = res['parent']['per_class']['tree']
    assert tree['total positives'] == direct['total positives']
    assert tree['total TP'] == direct['total TP']
    assert tree['total FP'] == direct['total FP']
    assert isclose(tree['AP'], direct['AP'])
    assert isclose(res['parent']['mAP'], direct['AP'])


def test_unmapped_classes_can_be_excluded_from_parent_view():
    gts = [_gt('A', 'maple', 1), _gt('A', 'rock', 3)]
    dets = [_det('A', 'rock', 0.7)]
    res = get_hierarchical_pascalvoc_metrics(gts, dets, MAP,
                                             iou_threshold=0.5,
                                             keep_unmapped=False)
    assert 'tree' in res['parent']['per_class']
    assert 'rock' not in res['parent']['per_class']
    # Leaf view still contains everything.
    assert 'rock' in res['leaf']['per_class']
