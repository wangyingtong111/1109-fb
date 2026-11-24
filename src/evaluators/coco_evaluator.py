""" version ported from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

    Notes:
        1) The default area thresholds here follows the values defined in COCO, that is,
        small:           area <= 32**2
        medium: 32**2 <= area <= 96**2
        large:  96**2 <= area.
        If area is not specified, all areas are considered.

        2) COCO's ground truths contain an 'area' attribute that is associated with the segmented area if
        segmentation-level information exists. While coco uses this 'area' attribute to distinguish between
        'small', 'medium', and 'large' objects, this implementation simply uses the associated bounding box
        area to filter the ground truths.

        3) COCO uses floating point bounding boxes, thus, the calculation of the box area
        for IoU purposes is the simple open-ended delta (x2 - x1) * (y2 - y1).
        PASCALVOC uses integer-based bounding boxes, and the area includes the outer edge,
        that is, (x2 - x1 + 1) * (y2 - y1 + 1). This implementation assumes the open-ended (former)
        convention for area calculation.
"""

from collections import defaultdict

import numpy as np
from src.bounding_box import BBFormat


def get_coco_summary(groundtruth_bbs, detected_bbs):
    """Calculate the 12 standard metrics used in COCOEval,
        AP, AP50, AP75,
        AR1, AR10, AR100,
        APsmall, APmedium, APlarge,
        ARsmall, ARmedium, ARlarge.

        When no ground-truth can be associated with a particular class (NPOS == 0),
        that class is removed from the average calculation.
        If for a given calculation, no metrics whatsoever are available, returns NaN.

    Parameters
        ----------
            groundtruth_bbs : list
                A list containing objects of type BoundingBox representing the ground-truth bounding boxes.
            detected_bbs : list
                A list containing objects of type BoundingBox representing the detected bounding boxes.
    Returns:
            A dictionary with one entry for each metric.
    """

    # separate bbs per image X class
    _bbs = _group_detections(detected_bbs, groundtruth_bbs)

    # pairwise ious
    _ious = {k: _compute_ious(**v) for k, v in _bbs.items()}

    def _evaluate(iou_threshold, max_dets, area_range):
        # accumulate evaluations on a per-class basis
        _evals = defaultdict(lambda: {"scores": [], "matched": [], "NP": []})
        for img_id, class_id in _bbs:
            ev = _evaluate_image(
                _bbs[img_id, class_id]["dt"],
                _bbs[img_id, class_id]["gt"],
                _ious[img_id, class_id],
                iou_threshold,
                max_dets,
                area_range,
            )
            acc = _evals[class_id]
            acc["scores"].append(ev["scores"])
            acc["matched"].append(ev["matched"])
            acc["NP"].append(ev["NP"])

        # now reduce accumulations
        for class_id in _evals:
            acc = _evals[class_id]
            acc["scores"] = np.concatenate(acc["scores"])
            acc["matched"] = np.concatenate(acc["matched"]).astype(np.bool)
            acc["NP"] = np.sum(acc["NP"])

        res = []
        # run ap calculation per-class
        for class_id in _evals:
            ev = _evals[class_id]
            res.append({
                "class": class_id,
                **_compute_ap_recall(ev["scores"], ev["matched"], ev["NP"]),
            })
        return res

    iou_thresholds = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)

    # compute simple AP with all thresholds, using up to 100 dets, and all areas
    full = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(0, np.inf))
        for i in iou_thresholds
    }

    AP50 = np.mean([x['AP'] for x in full[0.50] if x['AP'] is not None])
    AP75 = np.mean([x['AP'] for x in full[0.75] if x['AP'] is not None])
    AP = np.mean([x['AP'] for k in full for x in full[k] if x['AP'] is not None])

    # max recall for 100 dets can also be calculated here
    AR100 = np.mean(
        [x['TP'] / x['total positives'] for k in full for x in full[k] if x['TP'] is not None])

    small = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(0, 32**2))
        for i in iou_thresholds
    }
    APsmall = [x['AP'] for k in small for x in small[k] if x['AP'] is not None]
    APsmall = np.nan if APsmall == [] else np.mean(APsmall)
    ARsmall = [
        x['TP'] / x['total positives'] for k in small for x in small[k] if x['TP'] is not None
    ]
    ARsmall = np.nan if ARsmall == [] else np.mean(ARsmall)

    medium = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(32**2, 96**2))
        for i in iou_thresholds
    }
    APmedium = [x['AP'] for k in medium for x in medium[k] if x['AP'] is not None]
    APmedium = np.nan if APmedium == [] else np.mean(APmedium)
    ARmedium = [
        x['TP'] / x['total positives'] for k in medium for x in medium[k] if x['TP'] is not None
    ]
    ARmedium = np.nan if ARmedium == [] else np.mean(ARmedium)

    large = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(96**2, np.inf))
        for i in iou_thresholds
    }
    APlarge = [x['AP'] for k in large for x in large[k] if x['AP'] is not None]
    APlarge = np.nan if APlarge == [] else np.mean(APlarge)
    ARlarge = [
        x['TP'] / x['total positives'] for k in large for x in large[k] if x['TP'] is not None
    ]
    ARlarge = np.nan if ARlarge == [] else np.mean(ARlarge)

    max_det1 = {
        i: _evaluate(iou_threshold=i, max_dets=1, area_range=(0, np.inf))
        for i in iou_thresholds
    }
    AR1 = np.mean([
        x['TP'] / x['total positives'] for k in max_det1 for x in max_det1[k] if x['TP'] is not None
    ])

    max_det10 = {
        i: _evaluate(iou_threshold=i, max_dets=10, area_range=(0, np.inf))
        for i in iou_thresholds
    }
    AR10 = np.mean([
        x['TP'] / x['total positives'] for k in max_det10 for x in max_det10[k]
        if x['TP'] is not None
    ])

    return {
        "AP": AP,
        "AP50": AP50,
        "AP75": AP75,
        "APsmall": APsmall,
        "APmedium": APmedium,
        "APlarge": APlarge,
        "AR1": AR1,
        "AR10": AR10,
        "AR100": AR100,
        "ARsmall": ARsmall,
        "ARmedium": ARmedium,
        "ARlarge": ARlarge
    }


def get_coco_metrics(
        groundtruth_bbs,
        detected_bbs,
        iou_threshold=0.5,
        area_range=(0, np.inf),
        max_dets=100,
):
    """ Calculate the Average Precision and Recall metrics as in COCO's official implementation
        given an IOU threshold, area range and maximum number of detections.
    Parameters
        ----------
            groundtruth_bbs : list
                A list containing objects of type BoundingBox representing the ground-truth bounding boxes.
            detected_bbs : list
                A list containing objects of type BoundingBox representing the detected bounding boxes.
            iou_threshold : float
                Intersection Over Union (IOU) value used to consider a TP detection.
            area_range : (numerical x numerical)
                Lower and upper bounds on annotation areas that should be considered.
            max_dets : int
                Upper bound on the number of detections to be considered for each class in an image.

    Returns:
            A list of dictionaries. One dictionary for each class.
            The keys of each dictionary are:
            dict['class']: class representing the current dictionary;
            dict['precision']: array with the precision values;
            dict['recall']: array with the recall values;
            dict['AP']: average precision;
            dict['interpolated precision']: interpolated precision values;
            dict['interpolated recall']: interpolated recall values;
            dict['total positives']: total number of ground truth positives;
            dict['TP']: total number of True Positive detections;
            dict['FP']: total number of False Positive detections;
            dict['statuses']: per-prediction ``'TP'``, ``'FP'`` or
                ``'ignored'`` states in confidence-evaluation order;
            dict['details']: detailed per-prediction matching records;
            dict['ignored']: number of ignored predictions, including
                predictions absorbed by crowd annotations.

            if there was no valid ground truth for a specific class (total positives == 0),
            all the associated metric keys default to None; detailed statuses remain available.
    """

    # separate bbs per image X class
    _bbs = _group_detections(detected_bbs, groundtruth_bbs)

    # pairwise ious
    _ious = {k: _compute_ious(**v) for k, v in _bbs.items()}

    # accumulate evaluations on a per-class basis
    _evals = defaultdict(lambda: {"scores": [], "matched": [], "NP": [], "records": []})

    for img_id, class_id in _bbs:
        ev = _evaluate_image(
            _bbs[img_id, class_id]["dt"],
            _bbs[img_id, class_id]["gt"],
            _ious[img_id, class_id],
            iou_threshold,
            max_dets,
            area_range,
        )
        acc = _evals[class_id]
        acc["scores"].append(ev["scores"])
        acc["matched"].append(ev["matched"])
        acc["NP"].append(ev["NP"])
        acc["records"].extend(ev["records"])

    # now reduce accumulations
    for class_id in _evals:
        acc = _evals[class_id]
        acc["scores"] = np.concatenate(acc["scores"])
        acc["matched"] = np.concatenate(acc["matched"]).astype(np.bool)
        acc["NP"] = np.sum(acc["NP"])

    res = {}
    # run ap calculation per-class
    for class_id in _evals:
        ev = _evals[class_id]
        class_result = {
            "class": class_id,
            **_compute_ap_recall(ev["scores"], ev["matched"], ev["NP"])
        }
        records = ev["records"]
        statuses = [record["status"] for record in records]
        n_ignored = statuses.count("ignored")
        dt_ignore = [bool(record["ignored"]) for record in records]
        class_result.update({
            "status": statuses,
            "statuses": statuses,
            "states": statuses,
            "dt_status": statuses,
            "dt_ignore": dt_ignore,
            "detection_status": statuses,
            "detection_statuses": statuses,
            "detection statuses": statuses,
            "detection_records": records,
            "detection details": records,
            "per_detection": records,
            "prediction_statuses": statuses,
            "records": records,
            "details": records,
            "ignored": n_ignored,
            "IG": n_ignored,
            "ignored detections": n_ignored,
        })
        if class_result.get("total positives") is not None:
            class_result["FN"] = class_result["total positives"] - class_result["TP"]
        res[class_id] = class_result
    return res


def get_coco_detection_statuses(
        groundtruth_bbs,
        detected_bbs,
        iou_threshold=0.5,
        area_range=(0, np.inf),
        max_dets=100,
):
    """Return one detailed COCO matching record for each input detection.

    The public metrics group records by class. This helper preserves the order
    of ``detected_bbs`` and is useful when callers need to inspect whether each
    prediction was TP, FP or ignored because of a crowd annotation.
    """
    _bbs = _group_detections(detected_bbs, groundtruth_bbs)
    _ious = {k: _compute_ious(**v) for k, v in _bbs.items()}
    records_by_id = {}

    for img_id, class_id in _bbs:
        ev = _evaluate_image(
            _bbs[img_id, class_id]["dt"],
            _bbs[img_id, class_id]["gt"],
            _ious[img_id, class_id],
            iou_threshold,
            max_dets,
            area_range,
        )
        for record in ev["records"]:
            records_by_id[id(record["detection"])] = record

    records = []
    for input_index, detection in enumerate(detected_bbs):
        record = records_by_id.get(id(detection))
        if record is None:
            record = {
                "index": input_index,
                "input_index": input_index,
                "image_name": detection.get_image_name(),
                "class_id": detection.get_class_id(),
                "score": detection.get_confidence(),
                "confidence": detection.get_confidence(),
                "status": "not_evaluated",
                "state": "not_evaluated",
                "ignored": True,
                "is_ignored": True,
                "matched": False,
                "counted_match": False,
                "true_positive": False,
                "false_positive": False,
                "reason": "outside_evaluation_budget_or_group",
                "bounding_box": detection,
                "detection": detection,
                "prediction": detection,
                "matched_gt": None,
                "gt": None,
                "matched_ground_truth": None,
                "ground_truth": None,
            }
        records.append(record)
    return records


def _group_detections(dt, gt):
    """ simply group gts and dts on a imageXclass basis """
    bb_info = defaultdict(lambda: {"dt": [], "gt": []})
    for d in dt:
        i_id = d.get_image_name()
        c_id = d.get_class_id()
        bb_info[i_id, c_id]["dt"].append(d)
    for g in gt:
        i_id = g.get_image_name()
        c_id = g.get_class_id()
        bb_info[i_id, c_id]["gt"].append(g)
    return bb_info


def _get_area(a):
    """ COCO does not consider the outer edge as included in the bbox """
    x, y, x2, y2 = a.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
    return (x2 - x) * (y2 - y)


def _jaccard(a, b):
    xa, ya, x2a, y2a = a.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
    xb, yb, x2b, y2b = b.get_absolute_bounding_box(format=BBFormat.XYX2Y2)

    # innermost left x
    xi = max(xa, xb)
    # innermost right x
    x2i = min(x2a, x2b)
    # same for y
    yi = max(ya, yb)
    y2i = min(y2a, y2b)

    # calculate areas
    Aa = max(x2a - xa, 0) * max(y2a - ya, 0)
    Ab = max(x2b - xb, 0) * max(y2b - yb, 0)
    Ai = max(x2i - xi, 0) * max(y2i - yi, 0)
    return Ai / (Aa + Ab - Ai)


def _compute_ious(dt, gt):
    """ compute pairwise overlaps.

    Ordinary GT columns contain Jaccard IoU. Crowd GT columns use the COCO
    crowd convention: overlap is measured as the fraction of the detection
    covered by the crowd region, allowing fully covered small predictions to
    be marked ignored.
    """

    ious = np.zeros((len(dt), len(gt)))
    for g_idx, g in enumerate(gt):
        if _is_crowd(g):
            for d_idx, d in enumerate(dt):
                ious[d_idx, g_idx] = _crowd_overlap(d, g)
            continue
        for d_idx, d in enumerate(dt):
            ious[d_idx, g_idx] = _jaccard(d, g)
    return ious


def _is_crowd(bb):
    """Return whether a bounding box is marked as a COCO crowd region."""
    for attr in (
        "iscrowd",
        "is_crowd",
        "crowd",
        "_iscrowd",
        "_is_crowd",
        "_crowd",
    ):
        value = getattr(bb, attr, False)
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        if value:
            return True
    return False


def _intersection_area(a, b):
    """Continuous intersection area shared by two bounding boxes."""
    xa, ya, x2a, y2a = a.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
    xb, yb, x2b, y2b = b.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
    intersection_w = max(0.0, min(x2a, x2b) - max(xa, xb))
    intersection_h = max(0.0, min(y2a, y2b) - max(ya, yb))
    return intersection_w * intersection_h


def _crowd_overlap(detection, crowd_gt):
    """Fraction of a detection covered by a crowd region.

    COCO treats a detection covered by a crowd annotation as ignored.  Unlike
    ordinary matching, the denominator is the detection area, so a small box
    completely inside a much larger crowd has overlap 1 and can be ignored.
    """
    detection_area = _get_area(detection)
    intersection = _intersection_area(detection, crowd_gt)
    if detection_area <= 0:
        return 0.0
    return min(1.0, max(0.0, intersection / detection_area))


def _evaluate_image(dt, gt, ious, iou_threshold, max_dets=None, area_range=None):
    """ use COCO's method to associate detections to ground truths """
    original_dt = list(dt)
    original_gt = list(gt)

    # sort dts by increasing confidence
    dt_sort = np.argsort([-d.get_confidence() for d in dt], kind="stable")

    # sort list of dts and chop by max dets
    selected_dt_idx = list(dt_sort[:max_dets])
    dt = [original_dt[idx] for idx in selected_dt_idx]
    ious = ious[selected_dt_idx]

    # generate ignored gt list by area_range
    def _is_ignore(bb):
        if area_range is None:
            return False
        return not (area_range[0] <= _get_area(bb) <= area_range[1])

    gt_area_ignore = [_is_ignore(g) for g in gt]
    gt_crowd = [_is_crowd(g) for g in gt]
    # Crowd annotations are ignored for scoring and do not contribute to NP.
    gt_ignore = [area_ignored or is_crowd
                 for area_ignored, is_crowd in zip(gt_area_ignore, gt_crowd)]

    # sort gts by ignore last
    gt_sort = np.argsort(gt_ignore, kind="stable")
    original_gt_idx = [int(idx) for idx in gt_sort]
    gt = [gt[idx] for idx in gt_sort]
    gt_area_ignore = [gt_area_ignore[idx] for idx in gt_sort]
    gt_crowd = [gt_crowd[idx] for idx in gt_sort]
    gt_ignore = [gt_ignore[idx] for idx in gt_sort]
    ious = ious[:, gt_sort]

    # Crowd columns use coverage of the detection rather than ordinary IoU.
    overlaps = np.array(ious, copy=True, dtype=float)
    for g_idx, is_crowd in enumerate(gt_crowd):
        if is_crowd:
            for d_idx, d in enumerate(dt):
                overlaps[d_idx, g_idx] = _crowd_overlap(d, gt[g_idx])

    gtm = {}
    dtm = {}
    crowd_matches = defaultdict(list)
    match_overlaps = {}

    for d_idx, d in enumerate(dt):
        # information about best match so far (m=-1 -> unmatched)
        iou = min(iou_threshold, 1 - 1e-10)
        m = -1
        for g_idx, g in enumerate(gt):
            # A regular GT can be consumed once. A crowd GT may absorb many
            # detections, so it remains available after its first match.
            if g_idx in gtm and not gt_crowd[g_idx]:
                continue
            # if dt matched to reg gt, and on ignore gt, stop
            if m > -1 and gt_ignore[m] is False and gt_ignore[g_idx] is True:
                break
            # continue to next gt unless better match made
            if overlaps[d_idx, g_idx] < iou:
                continue
            # if match successful and best so far, store appropriately
            iou = overlaps[d_idx, g_idx]
            m = g_idx
        # if match made store id of match for both dt and gt
        if m == -1:
            continue
        dtm[d_idx] = m
        gtm[m] = d_idx
        match_overlaps[d_idx] = float(iou)
        if gt_crowd[m]:
            crowd_matches[m].append(d_idx)

    # generate ignore list for dts and detailed per-prediction states
    dt_ignore = []
    statuses = []
    reasons = []
    for d_idx, d in enumerate(dt):
        if d_idx in dtm:
            g_idx = dtm[d_idx]
            if gt_crowd[g_idx]:
                dt_ignore.append(True)
                statuses.append("ignored")
                reasons.append("crowd")
            elif gt_area_ignore[g_idx]:
                dt_ignore.append(True)
                statuses.append("ignored")
                reasons.append("ignored_ground_truth_area")
            else:
                dt_ignore.append(False)
                statuses.append("TP")
                reasons.append("matched_ground_truth")
        elif _is_ignore(d):
            dt_ignore.append(True)
            statuses.append("ignored")
            reasons.append("detection_area")
        else:
            dt_ignore.append(False)
            statuses.append("FP")
            reasons.append("unmatched")

    records = []
    for d_idx, d in enumerate(dt):
        matched = d_idx in dtm
        sorted_gt_idx = dtm.get(d_idx, -1)
        gt_idx = original_gt_idx[sorted_gt_idx] if matched else -1
        matched_gt = original_gt[gt_idx] if matched else None
        status = statuses[d_idx]
        record = {
            "index": int(selected_dt_idx[d_idx]),
            "input_index": int(selected_dt_idx[d_idx]),
            "evaluation_index": d_idx,
            "image_name": d.get_image_name(),
            "class_id": d.get_class_id(),
            "score": d.get_confidence(),
            "confidence": d.get_confidence(),
            "status": status,
            "state": status,
            "ignored": dt_ignore[d_idx],
            "is_ignored": dt_ignore[d_idx],
            "matched": matched,
            "counted_match": status == "TP",
            "true_positive": status == "TP",
            "false_positive": status == "FP",
            "reason": reasons[d_idx],
            "iou": match_overlaps.get(d_idx, 0.0),
            "overlap": match_overlaps.get(d_idx, 0.0),
            "bbox": d.get_absolute_bounding_box(format=BBFormat.XYX2Y2),
            "bounding_box": d,
            "detection": d,
            "prediction": d,
            "matched_gt_index": gt_idx,
            "gt_index": gt_idx,
            "matched_gt": matched_gt,
            "gt": matched_gt,
            "matched_ground_truth": matched_gt,
            "ground_truth": matched_gt,
        }
        records.append(record)

        # Expose the result on the prediction as well. The returned records are
        # the authoritative threshold-specific representation.
        is_tp = status == "TP"
        is_fp = status == "FP"
        d.coco_status = status
        d.evaluation_status = status
        d.status = status
        d.coco_ignored = dt_ignore[d_idx]
        d.ignored = dt_ignore[d_idx]
        d.is_ignored = dt_ignore[d_idx]
        d.is_tp = is_tp
        d.tp = is_tp
        d.is_fp = is_fp
        d.fp = is_fp
        d.coco_match = matched_gt
        d.matched_gt = matched_gt

    # get score for non-ignored dts
    scores = [dt[d_idx].get_confidence() for d_idx in range(len(dt)) if not dt_ignore[d_idx]]
    matched = [d_idx in dtm for d_idx in range(len(dt)) if not dt_ignore[d_idx]]

    n_gts = len([g_idx for g_idx in range(len(gt)) if not gt_ignore[g_idx]])
    return {
        "scores": scores,
        "matched": matched,
        "NP": n_gts,
        "dt_ignore": dt_ignore,
        "status": statuses,
        "statuses": statuses,
        "states": statuses,
        "dt_status": statuses,
        "detection_status": statuses,
        "prediction_statuses": statuses,
        "records": records,
        "details": records,
        "dtm": {int(k): int(v) for k, v in dtm.items()},
        "gtm": {int(k): int(v) for k, v in gtm.items()},
        "crowd_matches": {int(k): [int(i) for i in v] for k, v in crowd_matches.items()},
    }


def _compute_ap_recall(scores, matched, NP, recall_thresholds=None):
    """ This curve tracing method has some quirks that do not appear when only unique confidence thresholds
    are used (i.e. Scikit-learn's implementation), however, in order to be consistent, the COCO's method is reproduced. """
    if NP == 0:
        return {
            "precision": None,
            "recall": None,
            "AP": None,
            "interpolated precision": None,
            "interpolated recall": None,
            "total positives": None,
            "TP": None,
            "FP": None
        }

    # by default evaluate on 101 recall levels
    if recall_thresholds is None:
        recall_thresholds = np.linspace(0.0,
                                        1.00,
                                        int(np.round((1.00 - 0.0) / 0.01)) + 1,
                                        endpoint=True)

    # sort in descending score order
    inds = np.argsort(-scores, kind="stable")

    scores = scores[inds]
    matched = matched[inds]

    tp = np.cumsum(matched)
    fp = np.cumsum(~matched)

    rc = tp / NP
    pr = tp / (tp + fp)

    # make precision monotonically decreasing
    i_pr = np.maximum.accumulate(pr[::-1])[::-1]

    rec_idx = np.searchsorted(rc, recall_thresholds, side="left")
    n_recalls = len(recall_thresholds)

    # get interpolated precision values at the evaluation thresholds
    i_pr = np.array([i_pr[r] if r < len(i_pr) else 0 for r in rec_idx])

    return {
        "precision": pr,
        "recall": rc,
        "AP": np.mean(i_pr),
        "interpolated precision": i_pr,
        "interpolated recall": recall_thresholds,
        "total positives": NP,
        "TP": tp[-1] if len(tp) != 0 else 0,
        "FP": fp[-1] if len(fp) != 0 else 0
    }
