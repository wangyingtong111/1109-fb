"""Hierarchical evaluation of object detections.

This module adds a *hierarchical view* on top of the leaf-level (fine-grained)
evaluation performed by :mod:`src.evaluators.pascal_voc_evaluator`:

* The **leaf-level view** is computed exactly as before, on the original boxes
  and their original labels. The original lists of :class:`BoundingBox` are
  never modified, so they can keep being used for any other evaluation.
* The **parent-level view** first maps every leaf label to a parent label
  (e.g. ``maple``/``oak`` -> ``tree``) and then runs an *independent* matching
  and scoring pass. Both reports are returned simultaneously.

Entity-aware merging
--------------------
A physical entity may carry more than one leaf annotation with the very same
geometry (for instance, the same tree annotated once as ``maple`` and once as
``oak``, both mapped to parent ``tree``). Ground-truth boxes that share the
same ``(image_name, entity_id)`` and map to the same parent describe **one**
parent-class instance, so:

* they count as a single positive (the parent denominator is not inflated);
* all detections matching any of the sibling boxes compete for that single
  instance: the first one (highest confidence) scores one true positive and
  every further one is a false positive.

Entity identifiers are scoped per image: the same id used in another image
identifies an independent instance. Boxes without an ``entity_id`` are treated
as distinct entities, in which case the behaviour reduces to the standard
per-box matching of :func:`src.evaluators.pascal_voc_evaluator.get_pascalvoc_metrics`.
"""

from collections import Counter, OrderedDict

import numpy as np
import pandas as pd

from src.bounding_box import BoundingBox
from src.evaluators.pascal_voc_evaluator import (calculate_ap_11_point_interp,
                                                 calculate_ap_every_point,
                                                 get_pascalvoc_metrics)
from src.utils.enumerators import MethodAveragePrecision


def _parent_of(class_id, class_to_parent, keep_unmapped=True):
    """Map a leaf class id to its parent class id."""
    if class_id in class_to_parent:
        return class_to_parent[class_id]
    # A parent mapping may list parent classes as keys too; keep them as is.
    if class_id in class_to_parent.values():
        return class_id
    if keep_unmapped:
        return class_id
    return None


def map_bounding_boxes_to_parent(bounding_boxes, class_to_parent, keep_unmapped=True):
    """Return *clones* of the boxes whose class id is mapped to a parent class.

    The original boxes are left untouched, therefore their geometry and leaf
    labels remain available for other (fine-grained) evaluations.

    Parameters
    ----------
    bounding_boxes : list
        List of :class:`BoundingBox` objects.
    class_to_parent : dict
        Mapping ``{leaf_class_id: parent_class_id}``.
    keep_unmapped : bool
        If True (default), boxes whose class is absent from the mapping are
        cloned keeping their original class; if False, they are dropped.

    Returns
    -------
    list
        New list of cloned :class:`BoundingBox` objects carrying the parent
        class ids. Coordinates, confidence, bb type and entity id are preserved.
    """
    mapped = []
    for bb in bounding_boxes:
        parent = _parent_of(bb.get_class_id(), class_to_parent, keep_unmapped)
        if parent is None:
            continue
        clone = BoundingBox.clone(bb)
        clone.set_class_id(parent)
        mapped.append(clone)
    return mapped


def _entity_key(bb):
    """Identity of a parent-level ground-truth entity.

    Boxes without an explicit entity id are each their own entity, identified
    by their object identity so that coincident boxes never merge silently.
    """
    entity_id = bb.get_entity_id()
    if entity_id is None:
        return ('__no_entity__', id(bb))
    return (bb.get_image_name(), entity_id)


def get_parent_entities(gt_boxes, class_to_parent):
    """Group ground-truth boxes into parent-class entities.

    Parameters
    ----------
    gt_boxes : list
        List of ground-truth :class:`BoundingBox` objects (original leaf boxes).
    class_to_parent : dict
        Mapping ``{leaf_class_id: parent_class_id}``.

    Returns
    -------
    dict
        ``{parent_class: {entity_key: {'image': image_name, 'boxes': [leaf boxes]}}}``.
        The number of entity keys per parent equals the number of positives
        (denominator) used in the parent-level view, and the nested leaf boxes
        allow recomputing the leaf-level denominators as well.
    """
    entities = {}
    for bb in gt_boxes:
        parent = _parent_of(bb.get_class_id(), class_to_parent, keep_unmapped=True)
        entities.setdefault(parent, OrderedDict())
        key = _entity_key(bb)
        entities[parent].setdefault(key, {'image': bb.get_image_name(), 'boxes': []})
        entities[parent][key]['boxes'].append(bb)
    return entities


def _evaluate_parent_class(parent_class, gt_parent_boxes, det_parent_boxes, iou_threshold,
                           method, generate_table):
    """Run one independent parent-class matching/scoring pass.

    Ground-truth boxes belonging to the same ``(image, entity_id)`` are a
    single parent instance: they share one matched flag, so the set of
    predictions targeting that entity can contribute at most one TP.
    """
    # Group the (already parent-mapped) ground truths into entities.
    gt_groups = OrderedDict()
    for g in gt_parent_boxes:
        key = _entity_key(g)
        gt_groups.setdefault(key, []).append(g)

    # Stable per-image positional index of every group, mirroring the
    # bookkeeping arrays used by the leaf-level evaluator.
    image_to_group_indices = {}
    for idx, key in enumerate(gt_groups.keys()):
        image_name = gt_groups[key][0].get_image_name()
        image_to_group_indices.setdefault(image_name, []).append(idx)

    npos = len(gt_groups)
    group_matched = np.zeros(npos)

    # Sort detections by decreasing confidence (stable order for ties).
    dects = sorted(det_parent_boxes,
                   key=lambda bb: bb.get_confidence() if bb.get_confidence() is not None else 0.0,
                   reverse=True)
    TP = np.zeros(len(dects))
    FP = np.zeros(len(dects))

    dict_table = {
        'image': [],
        'confidence': [],
        'TP': [],
        'FP': [],
        'acc TP': [],
        'acc FP': [],
        'precision': [],
        'recall': []
    }

    gt_keys = list(gt_groups.keys())
    for idx_det, det in enumerate(dects):
        img_det = det.get_image_name()
        if generate_table:
            dict_table['image'].append(img_det)
            dict_table['confidence'].append(f'{100*det.get_confidence():.2f}%')

        iou_max = np.finfo(float).min
        match_group_idx = None
        # Consider every parent entity present in the image of the detection.
        for group_idx in image_to_group_indices.get(img_det, []):
            # Highest IoU against any sibling leaf box of the entity.
            group_iou = max(BoundingBox.iou(det, g) for g in gt_groups[gt_keys[group_idx]])
            if group_iou > iou_max:
                iou_max = group_iou
                match_group_idx = group_idx

        if match_group_idx is not None and iou_max >= iou_threshold and \
                group_matched[match_group_idx] == 0:
            TP[idx_det] = 1
            group_matched[match_group_idx] = 1
            if generate_table:
                dict_table['TP'].append(1)
                dict_table['FP'].append(0)
        else:
            # IoU below threshold, or the only overlapping parent entity was
            # already hit by an earlier (higher confidence) detection.
            FP[idx_det] = 1
            if generate_table:
                dict_table['FP'].append(1)
                dict_table['TP'].append(0)

    acc_FP = np.cumsum(FP)
    acc_TP = np.cumsum(TP)
    rec = acc_TP / npos
    prec = np.divide(acc_TP, (acc_FP + acc_TP))
    if generate_table:
        dict_table['acc TP'] = list(acc_TP)
        dict_table['acc FP'] = list(acc_FP)
        dict_table['precision'] = list(prec)
        dict_table['recall'] = list(rec)
        table = pd.DataFrame(dict_table)
    else:
        table = None

    if method == MethodAveragePrecision.EVERY_POINT_INTERPOLATION:
        ap, mpre, mrec, _ = calculate_ap_every_point(rec, prec)
    elif method == MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION:
        ap, mpre, mrec, _ = calculate_ap_11_point_interp(rec, prec)
    else:
        raise ValueError('method not defined')

    return {
        'precision': prec,
        'recall': rec,
        'AP': ap,
        'interpolated precision': mpre,
        'interpolated recall': mrec,
        'total positives': npos,
        'total TP': np.sum(TP),
        'total FP': np.sum(FP),
        'method': method,
        'iou': iou_threshold,
        'table': table
    }


def get_hierarchical_pascalvoc_metrics(gt_boxes,
                                       det_boxes,
                                       class_to_parent,
                                       iou_threshold=0.5,
                                       method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
                                       generate_table=False,
                                       keep_unmapped=True):
    """Compute leaf-level and parent-level Pascal VOC metrics at the same time.

    The two views are fully independent:

    * ``leaf`` contains the standard per-class report produced by
      :func:`get_pascalvoc_metrics` on the *original* boxes and labels, so its
      numbers are identical to running the leaf evaluation directly.
    * ``parent`` contains the report after mapping every label to its parent
      class, with sibling leaf annotations of the same physical entity merged
      into one positive and predictions on that entity capped at one TP.

    Parameters
    ----------
    gt_boxes : list
        Ground-truth :class:`BoundingBox` objects. They are not modified.
    det_boxes : list
        Detected :class:`BoundingBox` objects. They are not modified.
    class_to_parent : dict
        Mapping ``{leaf_class_id: parent_class_id}`` (e.g.
        ``{'maple': 'tree', 'oak': 'tree'}``). Parent classes may also appear
        as keys mapped to themselves.
    iou_threshold : float
        IoU threshold for a detection to be considered a match.
    method : MethodAveragePrecision
        AP interpolation method, applied identically to both views.
    generate_table : bool
        If True, per-detection TP/FP tables are built for both views.
    keep_unmapped : bool
        If True (default), classes absent from the mapping keep their own
        (leaf) label in the parent view; if False, they are excluded from it.

    Returns
    -------
    dict
        ``{'leaf': {'per_class': ..., 'mAP': ...},
           'parent': {'per_class': ..., 'mAP': ..., 'entities': ...}}``.
        Each ``per_class`` entry has the same structure as the dictionaries
        returned by :func:`get_pascalvoc_metrics`, so the existing plotting
        helpers work unchanged. ``entities`` documents the parent-level
        positives and their underlying leaf boxes, allowing the denominators
        of both views to be recomputed/audited.
    """
    # ----- Leaf-level view: original boxes, original labels, unchanged logic.
    leaf_results = get_pascalvoc_metrics(gt_boxes,
                                         det_boxes,
                                         iou_threshold=iou_threshold,
                                         method=method,
                                         generate_table=generate_table)

    # ----- Parent-level view: work on clones so original boxes stay intact.
    gt_parent = map_bounding_boxes_to_parent(gt_boxes, class_to_parent, keep_unmapped)
    det_parent = map_bounding_boxes_to_parent(det_boxes, class_to_parent, keep_unmapped)

    classes_bbs = {}
    gt_parent_classes = []
    for bb in gt_parent:
        c = bb.get_class_id()
        gt_parent_classes.append(c)
        classes_bbs.setdefault(c, {'gt': [], 'det': []})
        classes_bbs[c]['gt'].append(bb)
    for bb in det_parent:
        c = bb.get_class_id()
        classes_bbs.setdefault(c, {'gt': [], 'det': []})
        classes_bbs[c]['det'].append(bb)
    gt_parent_classes = list(set(gt_parent_classes))

    parent_ret = {}
    for c, v in classes_bbs.items():
        # Report only parent classes that occur in the ground truth.
        if c not in gt_parent_classes:
            continue
        parent_ret[c] = _evaluate_parent_class(c,
                                               v['gt'],
                                               v['det'],
                                               iou_threshold=iou_threshold,
                                               method=method,
                                               generate_table=generate_table)

    mAP = (sum(v['AP'] for k, v in parent_ret.items() if k in gt_parent_classes) /
           len(gt_parent_classes)) if gt_parent_classes else 0.0

    # Positives of both views, kept explicit so denominators can be recomputed.
    entities = get_parent_entities(gt_boxes, class_to_parent)
    parent_entities = {
        parent: {
            'total positives': len(group),
            'entities': [{
                'image': info['image'],
                'entity_id': key[1] if key[0] != '__no_entity__' else None,
                'leaf classes': sorted({bb.get_class_id() for bb in info['boxes']}),
                'leaf boxes': len(info['boxes'])
            } for key, info in group.items()]
        }
        for parent, group in entities.items()
        if parent in gt_parent_classes
    }
    leaf_positives = Counter(bb.get_class_id() for bb in gt_boxes)

    return {
        'leaf': leaf_results,
        'parent': {
            'per_class': parent_ret,
            'mAP': mAP,
            'entities': parent_entities,
            'leaf positives': dict(leaf_positives),
            'class_to_parent': dict(class_to_parent),
            'iou': iou_threshold
        }
    }
