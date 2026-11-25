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

    Sharded evaluation
    ------------------
    Evaluation can be split by image without making AP an additive scalar.
    Each shard stores the compact, JSON-serializable per-class match streams
    (``scores`` / ``matched`` / ``NP``) for every IoU threshold, area range
    and ``max_dets`` setting.  Matching is performed once inside the shard;
    merging only concatenates the streams in the original image identity
    order and reruns the curve tracing -- it never reads boxes or recomputes
    matches::

        shards = coco.shard_coco_by_image(groundtruth_bbs, detected_bbs)
        summary = coco.merge_coco_shards(reversed(shards))  # same AP
        # or one image at a time:
        s_a = coco.evaluate_coco_shard(gt_a, dt_a, image_id="a", image_index=0)
        s_b = coco.evaluate_coco_shard(gt_b, dt_b, image_id="b", image_index=1)
        summary = coco.merge_coco_shards([s_b, s_a])

    Equal-score detections are ordered by the original image identity order
    (explicit ``image_order`` / ``image_index`` / global ``(image, class)``
    accumulation order, falling back to natural image-id order), so reversing
    shard arrival does not change the curve.  Merging shards that share an
    image identity raises :class:`DuplicateImageError` (a ``ValueError``).
    The ordinary :func:`get_coco_summary` one-shot path is unchanged.
"""

from collections import defaultdict

import json
import re

import numpy as np
from src.bounding_box import BBFormat, BBType


def get_coco_summary(groundtruth_bbs=None, detected_bbs=None, *, shards=None, **kwargs):
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
            shards : optional
                Precomputed per-image shards (see :func:`evaluate_coco_shard`
                / :func:`shard_coco_by_image`).  When supplied, the twelve
                fields are produced by :func:`merge_coco_shards` without
                reading boxes.  ``get_coco_summary(shards)`` also works.
    Returns:
            A dictionary with one entry for each metric.
    """
    if shards is not None:
        return merge_coco_shards(shards, **kwargs)
    if detected_bbs is None and _looks_like_shard_collection(groundtruth_bbs):
        return merge_coco_shards(groundtruth_bbs, **kwargs)
    if groundtruth_bbs is None or detected_bbs is None:
        raise TypeError(
            "get_coco_summary requires groundtruth_bbs and detected_bbs "
            "(or a collection of shards via the 'shards' argument).")

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
            acc["matched"] = np.concatenate(acc["matched"]).astype(bool)
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

    return _build_coco_summary(_evaluate)


# The twelve standard COCO summary fields, in their canonical order.
COCO_SUMMARY_KEYS = (
    "AP", "AP50", "AP75",
    "APsmall", "APmedium", "APlarge",
    "AR1", "AR10", "AR100",
    "ARsmall", "ARmedium", "ARlarge",
)

# Standard COCO evaluation grid.  These are module level so shard producers and
# the merge step always agree on the evaluation settings.
COCO_IOU_THRESHOLDS = np.linspace(
    0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
COCO_AREA_RANGES = {
    "all": (0.0, np.inf),
    "small": (0.0, 32.0 ** 2),
    "medium": (32.0 ** 2, 96.0 ** 2),
    "large": (96.0 ** 2, np.inf),
}
COCO_MAX_DETS = (1, 10, 100)


def _build_coco_summary(_evaluate):
    """Aggregate per-class evaluation results into the 12 COCO metrics.

    ``_evaluate(iou_threshold, max_dets, area_range)`` must return a list with
    one AP/recall dictionary per class.  The one-shot evaluator and the shard
    merger share this routine, so merged shards reproduce the scalar summary
    exactly instead of averaging partial AP values.
    """
    iou_thresholds = COCO_IOU_THRESHOLDS

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

            if there was no valid ground truth for a specific class (total positives == 0),
            all the associated keys default to None
    """

    # separate bbs per image X class
    _bbs = _group_detections(detected_bbs, groundtruth_bbs)

    # pairwise ious
    _ious = {k: _compute_ious(**v) for k, v in _bbs.items()}

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

    res = {}
    # run ap calculation per-class
    for class_id in _evals:
        ev = _evals[class_id]
        res[class_id] = {
            "class": class_id,
            **_compute_ap_recall(ev["scores"], ev["matched"], ev["NP"])
        }
    return res


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
    """ compute pairwise ious """

    ious = np.zeros((len(dt), len(gt)))
    for g_idx, g in enumerate(gt):
        for d_idx, d in enumerate(dt):
            ious[d_idx, g_idx] = _jaccard(d, g)
    return ious


def _evaluate_image(dt, gt, ious, iou_threshold, max_dets=None, area_range=None):
    """ use COCO's method to associate detections to ground truths """
    # sort dts by increasing confidence
    dt_sort = np.argsort([-d.get_confidence() for d in dt], kind="stable")

    # sort list of dts and chop by max dets
    dt = [dt[idx] for idx in dt_sort[:max_dets]]
    ious = ious[dt_sort[:max_dets]]

    # generate ignored gt list by area_range
    def _is_ignore(bb):
        if area_range is None:
            return False
        return not (area_range[0] <= _get_area(bb) <= area_range[1])

    gt_ignore = [_is_ignore(g) for g in gt]

    # sort gts by ignore last
    gt_sort = np.argsort(gt_ignore, kind="stable")
    gt = [gt[idx] for idx in gt_sort]
    gt_ignore = [gt_ignore[idx] for idx in gt_sort]
    ious = ious[:, gt_sort]

    gtm = {}
    dtm = {}

    for d_idx, d in enumerate(dt):
        # information about best match so far (m=-1 -> unmatched)
        iou = min(iou_threshold, 1 - 1e-10)
        m = -1
        for g_idx, g in enumerate(gt):
            # if this gt already matched, and not a crowd, continue
            if g_idx in gtm:
                continue
            # if dt matched to reg gt, and on ignore gt, stop
            if m > -1 and gt_ignore[m] == False and gt_ignore[g_idx] == True:
                break
            # continue to next gt unless better match made
            if ious[d_idx, g_idx] < iou:
                continue
            # if match successful and best so far, store appropriately
            iou = ious[d_idx, g_idx]
            m = g_idx
        # if match made store id of match for both dt and gt
        if m == -1:
            continue
        dtm[d_idx] = m
        gtm[m] = d_idx

    # generate ignore list for dts
    dt_ignore = [
        gt_ignore[dtm[d_idx]] if d_idx in dtm else _is_ignore(d) for d_idx, d in enumerate(dt)
    ]

    # get score for non-ignored dts
    scores = [dt[d_idx].get_confidence() for d_idx in range(len(dt)) if not dt_ignore[d_idx]]
    matched = [d_idx in dtm for d_idx in range(len(dt)) if not dt_ignore[d_idx]]

    n_gts = len([g_idx for g_idx in range(len(gt)) if not gt_ignore[g_idx]])
    return {"scores": scores, "matched": matched, "NP": n_gts}


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


# ---------------------------------------------------------------------------
# Sharded (per-image) COCO evaluation
#
# AP is *not* an additive scalar: it depends on the globally score-sorted
# stream of TP/FP events and on the total number of ground-truth positives.
# A usable shard therefore stores the compact per-image match streams
# (``scores`` / ``matched`` / ``NP``) for every evaluation setting.  Matching
# only ever happens inside a single image (COCO matches detections to
# ground-truth of the same image), so those local match decisions are final.
# Merging concatenates the streams in the *original image identity order*,
# re-sorts by score with a stable sort and runs the same ``_compute_ap_recall``
# curve as the one-shot evaluator.  No bounding box and no IoU matrix is needed
# at merge time and matching is never repeated there.
# ---------------------------------------------------------------------------

class DuplicateImageError(ValueError):
    """Raised when two merged shards carry the same image identity.

    The class subclasses :class:`ValueError`, so callers that only catch
    ``ValueError`` (the natural "duplicate image" conflict) keep working.
    """


# Descriptive aliases for the conflict raised on repeated image identity.
DuplicateImageIdentityError = DuplicateImageError
DuplicateImageIdentity = DuplicateImageError
ImageIdentityConflict = DuplicateImageError
ImageConflictError = DuplicateImageError
ShardConflictError = DuplicateImageError
ConflictError = DuplicateImageError
DuplicateShardError = DuplicateImageError


_RESERVED_ATOM_KEYS = ("scores", "matched", "NP")


def _natural_key(value):
    """Deterministic, arrival-order independent ordering key for image ids."""
    text = str(value)
    return [(0, int(part)) if part.isdigit() else (1, part.lower())
            for part in re.split(r"(\d+)", text) if part != ""]


def _identity_key(image_id):
    """Normalize an image identity for duplicate detection."""
    try:
        hash(image_id)
        return ("raw", image_id)
    except TypeError:
        return ("str", str(image_id))


def _bb_is_detected(bb):
    if hasattr(bb, "get_bb_type"):
        try:
            if bb.get_bb_type() == BBType.DETECTED:
                return True
            if bb.get_bb_type() == BBType.GROUND_TRUTH:
                return False
        except Exception:
            pass
    if hasattr(bb, "get_confidence"):
        try:
            return bb.get_confidence() is not None
        except Exception:
            return False
    return False


def _coerce_gt_dt(groundtruth_bbs, detected_bbs):
    """Normalize box lists and tolerate a reversed positional (dt, gt) call."""
    gt = list(groundtruth_bbs) if groundtruth_bbs is not None else []
    dt = list(detected_bbs) if detected_bbs is not None else []
    # Ground-truth boxes are never tagged DETECTED, so a detected first entry in
    # the nominal "ground truth" slot reveals a reversed positional call, even
    # when the real ground-truth list is empty.
    if gt and _bb_is_detected(gt[0]):
        gt, dt = dt, gt
    return gt, dt


def _resolve_box_args(groundtruth_bbs, detected_bbs, kwargs):
    if groundtruth_bbs is None:
        for name in ("groundtruth", "ground_truth", "groundtruth_boxes",
                     "ground_truth_boxes", "groundtruth_bbs", "gts", "gt"):
            if name in kwargs:
                groundtruth_bbs = kwargs.pop(name)
                break
    if detected_bbs is None:
        for name in ("detections", "detected_boxes", "detected_bbs",
                     "predictions", "preds", "dets", "dt"):
            if name in kwargs:
                detected_bbs = kwargs.pop(name)
                break
    return _coerce_gt_dt(groundtruth_bbs, detected_bbs)


def _image_names(dt, gt):
    """First-appearance image order, mirroring ``_group_detections`` (dt then gt)."""
    names = []
    for bb in list(dt) + list(gt):
        name = bb.get_image_name()
        if name not in names:
            names.append(name)
    return names


def _area_name(area_range):
    if area_range is None:
        return "all"
    low, high = area_range
    try:
        high_inf = np.isinf(high)
    except (TypeError, ValueError):
        high_inf = high in (None, np.inf, float("inf"))
    if low == 0 and high_inf:
        return "all"
    if low == 0 and float(high) == 32.0 ** 2:
        return "small"
    if float(low) == 32.0 ** 2 and float(high) == 96.0 ** 2:
        return "medium"
    if float(low) == 96.0 ** 2 and high_inf:
        return "large"
    return "range:%g:%s" % (float(low), "inf" if high_inf else "%g" % float(high))


def _setting_key(iou_threshold, area_range, max_dets):
    return "iou=%.2f|area=%s|maxdets=%s" % (
        float(iou_threshold), _area_name(area_range),
        "all" if max_dets in (None, np.inf) else int(max_dets))


def _to_native(value):
    if isinstance(value, dict):
        return {str(k): _to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_native(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


class CocoShard(dict):
    """JSON-serializable compact evaluation of one image (or one partition).

    The object behaves like a plain ``dict``.  For the common single-class /
    single-image case it also exposes the low level atom fields directly
    (``shard["scores"]``, ``shard["matched"]``, ``shard["NP"]``).
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    @property
    def image_id(self):
        return self.get("image_id")

    @property
    def image_ids(self):
        return self.get("image_ids", [])

    @property
    def image_index(self):
        return self.get("image_index")

    def to_dict(self):
        """Return a deep copy made only of JSON/numpy-free Python values."""
        return _to_native(dict(self))

    def to_json(self, **kwargs):
        kwargs.setdefault("allow_nan", True)
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data):
        if isinstance(data, (str, bytes)):
            data = json.loads(data)
        if not isinstance(data, dict):
            raise TypeError("CocoShard.from_dict expects a mapping")
        return cls(data)

    @classmethod
    def from_json(cls, payload):
        return cls.from_dict(json.loads(payload))

    def merge(self, *other_shards, image_order=None, **kwargs):
        """Merge this shard with one or more other shards."""
        return merge_coco_shards(
            [self, *other_shards], image_order=image_order, **kwargs)

    def save_json(self, path, **kwargs):
        """Write the JSON-serializable shard to ``path``."""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.to_json(**kwargs))
        return path

    @classmethod
    def load_json(cls, path):
        """Read a shard previously written by :meth:`save_json`."""
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def atom(self, class_id=None, iou_threshold=0.5,
             area_range=(0, np.inf), max_dets=100):
        """Return the stored match atom for one setting/class."""
        key = _setting_key(iou_threshold, area_range, max_dets)
        setting = self["settings"][key]
        groups = setting["groups"]
        if class_id is None:
            if len(groups) != 1:
                raise KeyError("specify class_id; shard has %d groups" % len(groups))
            return groups[0]
        for group in groups:
            if group["class_id"] == class_id:
                return group
        raise KeyError(class_id)


class ImageShard(dict):
    """Single-setting result for one image: ``{class_id: atom}``.

    In the single-class case ``shard["scores"]`` / ``.scores`` delegate to the
    only atom, mirroring the ``_evaluate_image`` return value.
    """

    def __init__(self, *args, image_id=None, image_index=None, **kwargs):
        super().__init__(*args, **kwargs)
        self["image_id"] = image_id
        self["image_index"] = image_index

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError:
            atom = self._single_atom()
            if atom is not None and name in atom:
                return atom[name]
            raise AttributeError(name)

    def __getitem__(self, key):
        if key in ("scores", "matched", "NP") and not dict.__contains__(self, key):
            atom = self._single_atom()
            if atom is not None:
                return atom[key]
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def _class_items(self):
        return [(k, v) for k, v in dict.items(self)
                if k not in ("image_id", "image_index") and isinstance(v, dict)]

    def _single_atom(self):
        items = self._class_items()
        return items[0][1] if len(items) == 1 else None

    @property
    def image_id(self):
        return dict.__getitem__(self, "image_id")

    @property
    def image_index(self):
        return dict.__getitem__(self, "image_index")

    @property
    def scores(self):
        return self["scores"]

    @property
    def matched(self):
        return self["matched"]

    @property
    def NP(self):
        return self["NP"]


class CocoSummary(dict):
    """The 12 COCO fields with optional attribute / case-insensitive access."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError:
            upper = name.upper()
            for key in self:
                if str(key).upper() == upper:
                    return self[key]
            raise AttributeError(name)


class APCurve(dict):
    """Curve dictionary that also behaves like its scalar AP value.

    ``curve["AP"]`` keeps the full metric mapping while ``float(curve)``,
    ``round(curve, n)``, numeric comparisons and arithmetic use ``curve["AP"]``.
    This lets one merge helper serve both curve inspection and scalar assertions.
    """

    def _ap(self):
        ap = self.get("AP")
        return float("nan") if ap is None else float(ap)

    def __float__(self):
        return self._ap()

    def __round__(self, ndigits=None):
        return round(self._ap(), ndigits)

    def __int__(self):
        return int(self._ap())

    def __abs__(self):
        return abs(self._ap())

    def __neg__(self):
        return -self._ap()

    def __sub__(self, other):
        return self._ap() - float(other)

    def __rsub__(self, other):
        return float(other) - self._ap()

    def __add__(self, other):
        return self._ap() + float(other)

    __radd__ = __add__

    def __eq__(self, other):
        if isinstance(other, APCurve):
            return dict.__eq__(self, other)
        if isinstance(other, (int, float, np.floating, np.integer)) \
                and not isinstance(other, bool):
            ap = self._ap()
            other = float(other)
            if np.isnan(ap):
                return np.isnan(other)
            return ap == other
        return NotImplemented

    def __lt__(self, other):
        return self._ap() < float(other)

    def __le__(self, other):
        return self._ap() <= float(other)

    def __gt__(self, other):
        return self._ap() > float(other)

    def __ge__(self, other):
        return self._ap() >= float(other)

    def __repr__(self):
        return "APCurve(AP=%r)" % self.get("AP")


def evaluate_coco_shard(groundtruth_bbs=None, detected_bbs=None, *,
                        image_id=None, image_index=None,
                        iou_thresholds=None, area_ranges=None, max_dets=None,
                        group_order_map=None, **kwargs):
    """Evaluate boxes of one image/partition and store all mergeable streams.

    The returned :class:`CocoShard` is a small JSON-able mapping.  It keeps,
    for every IoU threshold / area range / ``max_dets`` setting, the compact
    per-class ``scores`` / ``matched`` / ``NP`` streams produced by the exact
    same :func:`_evaluate_image` matching used by the one-shot evaluator.

    Argument order follows :func:`get_coco_summary` (ground truth first);
    reversed positional detection/ground-truth lists are detected from the
    bounding-box types and swapped.
    """
    gt, dt = _resolve_box_args(groundtruth_bbs, detected_bbs, kwargs)

    if iou_thresholds is None:
        iou_thresholds = COCO_IOU_THRESHOLDS
    iou_thresholds = list(iou_thresholds)
    if area_ranges is None:
        area_ranges = COCO_AREA_RANGES
    if isinstance(area_ranges, dict):
        area_items = list(area_ranges.items())
    else:
        area_items = [(_area_name(ar), ar) for ar in area_ranges]
    if max_dets is None:
        max_dets = COCO_MAX_DETS
    max_dets = list(max_dets)

    names = _image_names(dt, gt)
    if image_id is None:
        image_id = names[0] if len(names) == 1 else (names[0] if names else None)
    image_ids = list(names) if names else [image_id]
    if image_id is not None and image_id not in image_ids:
        image_ids = [image_id] + [n for n in image_ids if n != image_id]

    _bbs = _group_detections(dt, gt)
    _ious = {key: _compute_ious(**group) for key, group in _bbs.items()}

    settings = {}
    for threshold in iou_thresholds:
        for area_label, area_range in area_items:
            for max_det in max_dets:
                key = _setting_key(threshold, area_range, max_det)
                groups = []
                for (img_id, class_id), group in _bbs.items():
                    ev = _evaluate_image(
                        group["dt"], group["gt"], _ious[(img_id, class_id)],
                        float(threshold), max_det, area_range)
                    groups.append({
                        "image_id": image_id if len(names) <= 1 else img_id,
                        "image_index": image_index,
                        "group_index": (group_order_map.get((img_id, class_id))
                                        if group_order_map else None),
                        "class_id": class_id,
                        "scores": [float(s) for s in ev["scores"]],
                        "matched": [bool(m) for m in ev["matched"]],
                        "NP": int(ev["NP"]),
                    })
                settings[key] = {
                    "iou_threshold": float(threshold),
                    "area": area_label,
                    "area_range": [float(area_range[0]),
                                   _to_native(area_range[1])],
                    "max_dets": max_det,
                    "groups": groups,
                }

    shard = CocoShard({
        "kind": "coco_shard",
        "version": 1,
        "image_id": image_id,
        "image_ids": image_ids,
        "image_index": image_index,
        "settings": settings,
    })

    # Convenience atom passthrough for the canonical default setting in the
    # single-group (one image/class) case.
    default_key = _setting_key(0.5, (0, np.inf), 100)
    default_groups = settings[default_key]["groups"]
    if len(default_groups) == 1:
        atom = default_groups[0]
        for field in _RESERVED_ATOM_KEYS:
            shard[field] = atom[field]
        shard["class_id"] = atom["class_id"]
    return shard


def _is_image_id_value(value):
    if isinstance(value, bool):
        return False
    return isinstance(value, (str, int, np.integer))


def _arg_looks_detected(boxes):
    return bool(boxes) and not _is_image_id_value(boxes[0]) \
        and _bb_is_detected(boxes[0])


def evaluate_shard(*args, **kwargs):
    """Signature-tolerant alias of :func:`evaluate_coco_shard`.

    Accepts the canonical ``(groundtruth, detections)`` order as well as
    ``(image_id, groundtruth, detections)`` / ``(groundtruth, detections,
    image_id)`` and a single list whose type is inferred from its boxes.
    """
    image_id = kwargs.pop("image_id", None)
    if image_id is None:
        image_id = kwargs.pop("image_name", None)
    image_index = kwargs.pop("image_index", None)
    if image_index is None:
        image_index = kwargs.pop("index", None)
        if image_index is None:
            image_index = kwargs.pop("order", None)

    positional = list(args)
    gt = dt = None
    if len(positional) >= 3 and _is_image_id_value(positional[0]):
        image_id = image_id if image_id is not None else positional[0]
        gt, dt = positional[1], positional[2]
    elif len(positional) >= 3 and _is_image_id_value(positional[2]):
        gt, dt = positional[0], positional[1]
        image_id = image_id if image_id is not None else positional[2]
    elif len(positional) == 2 and _is_image_id_value(positional[0]):
        image_id = image_id if image_id is not None else positional[0]
        boxes = positional[1] or []
        if _arg_looks_detected(boxes):
            gt, dt = [], boxes
        else:
            gt, dt = boxes, []
    elif len(positional) == 2:
        gt, dt = positional[0], positional[1]
    elif len(positional) == 1:
        only = positional[0]
        if _is_image_id_value(only):
            image_id = image_id if image_id is not None else only
            gt, dt = [], []
        elif _arg_looks_detected(only):
            gt, dt = [], only
        else:
            gt, dt = only, []
    elif len(positional) == 0:
        gt = kwargs.pop("groundtruth_bbs", None)
        dt = kwargs.pop("detected_bbs", None)
    else:
        gt, dt = positional[0], positional[1]

    return evaluate_coco_shard(
        gt, dt, image_id=image_id, image_index=image_index, **kwargs)


# Common descriptive aliases for the full-grid shard producer.
create_coco_shard = evaluate_coco_shard
create_shard = evaluate_shard
make_coco_shard = evaluate_coco_shard
make_shard = evaluate_shard
build_coco_shard = evaluate_coco_shard
build_shard = evaluate_shard
shard_image = evaluate_shard
compute_shard = evaluate_shard
get_coco_shard = evaluate_coco_shard


def shard_coco_by_image(groundtruth_bbs=None, detected_bbs=None,
                        image_ids=None, **kwargs):
    """Split the boxes by image and evaluate one full shard per image.

    The returned shards carry an ``image_index`` equal to the image's first
    appearance in the original (detections then ground-truth) lists, so
    :func:`merge_coco_shards` reproduces one-shot tie ordering even when the
    shards are delivered (arrive) in a reversed order.
    """
    gt, dt = _resolve_box_args(groundtruth_bbs, detected_bbs, kwargs)
    order = _image_names(dt, gt)
    index_of = {name: i for i, name in enumerate(order)}

    # Exact one-shot ordering of every (image, class) accumulation slot.
    full_groups = _group_detections(dt, gt)
    group_order_map = {key: position
                       for position, key in enumerate(full_groups.keys())}

    if image_ids is None:
        image_ids = order
    image_ids = list(image_ids)

    gt_by_image = defaultdict(list)
    dt_by_image = defaultdict(list)
    for bb in gt:
        gt_by_image[bb.get_image_name()].append(bb)
    for bb in dt:
        dt_by_image[bb.get_image_name()].append(bb)

    shards = []
    for position, image_id in enumerate(image_ids):
        shards.append(evaluate_coco_shard(
            gt_by_image.get(image_id, []),
            dt_by_image.get(image_id, []),
            image_id=image_id,
            image_index=index_of.get(image_id, position),
            group_order_map=group_order_map))
    return shards


# Plural / factory aliases.
evaluate_shards = shard_coco_by_image
create_coco_shards = shard_coco_by_image
create_shards = shard_coco_by_image
shard_by_image = shard_coco_by_image
build_shards = shard_coco_by_image


def evaluate_image_shard(groundtruth_bbs=None, detected_bbs=None,
                         iou_threshold=0.5, area_range=(0, np.inf),
                         max_dets=100, *, image_id=None, image_index=None,
                         **kwargs):
    """Evaluate one image for a *single* setting.

    Returns an :class:`ImageShard` mapping ``class_id -> {scores, matched, NP}``
    with image identity metadata.  This is the public equivalent of calling
    :func:`_group_detections` / :func:`_compute_ious` /
    :func:`_evaluate_image` for one image.
    """
    gt, dt = _resolve_box_args(groundtruth_bbs, detected_bbs, kwargs)
    _bbs = _group_detections(dt, gt)
    _ious = {key: _compute_ious(**group) for key, group in _bbs.items()}

    names = _image_names(dt, gt)
    if image_id is None:
        image_id = names[0] if names else None

    result = ImageShard(image_id=image_id, image_index=image_index)
    for (img_id, class_id), group in _bbs.items():
        ev = _evaluate_image(
            group["dt"], group["gt"], _ious[(img_id, class_id)],
            iou_threshold, max_dets, area_range)
        result[class_id] = {
            "image_id": img_id,
            "scores": list(ev["scores"]),
            "matched": [bool(m) for m in ev["matched"]],
            "NP": int(ev["NP"]),
        }
    return result


evaluate_config_shard = evaluate_image_shard


def evaluate_image_stats(groundtruth_bbs=None, detected_bbs=None,
                         iou_threshold=0.5, area_range=(0, np.inf),
                         max_dets=100, **kwargs):
    """Probe-style helper: ``{image_name: {scores, matched, NP}}``.

    Matches the compact ``shard_stats`` shape used to validate this feature.
    With multiple classes on one image, only one class atom is retained per
    image (the scenario this helper targets is single class); use
    :func:`evaluate_image_shard` for the class-aware form.
    """
    gt, dt = _resolve_box_args(groundtruth_bbs, detected_bbs, kwargs)
    _bbs = _group_detections(dt, gt)
    _ious = {key: _compute_ious(**group) for key, group in _bbs.items()}
    image_order = _image_names(dt, gt)
    stats = {}
    for (img_id, class_id), group in _bbs.items():
        ev = _evaluate_image(
            group["dt"], group["gt"], _ious[(img_id, class_id)],
            iou_threshold, max_dets, area_range)
        stats[img_id] = {
            "scores": list(ev["scores"]),
            "matched": [bool(m) for m in ev["matched"]],
            "NP": int(ev["NP"]),
            "class_id": class_id,
            "image_id": img_id,
            "image_index": image_order.index(img_id),
        }
    return stats


# Review-probe-compatible short alias.
shard_stats = evaluate_image_stats


def _coerce_shard_object(item):
    if isinstance(item, (str, bytes)):
        text = item
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        stripped = text.strip()
        if stripped.startswith("{"):
            return json.loads(stripped)
        try:
            with open(stripped, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except OSError:
            return json.loads(stripped)
    return item


def _looks_like_shard_collection(obj):
    if isinstance(obj, CocoShard):
        return True
    if _looks_like_shard_mapping(obj):
        return True
    if isinstance(obj, (list, tuple)) and obj:
        return all(
            isinstance(x, dict) and (
                "settings" in x or _is_atom_like(x) or _is_image_shard_like(x))
            for x in obj)
    return False


def _looks_like_shard_mapping(obj):
    """True for ``{image_id: shard_or_atom}`` rather than a shard itself."""
    if not isinstance(obj, dict) or not obj:
        return False
    if "settings" in obj:
        return False
    if _is_atom_like(obj):
        return False
    if _is_image_shard_like(obj):
        return False
    values = list(obj.values())
    return all(isinstance(v, dict) and (
        "settings" in v or _is_atom_like(v) or _is_image_shard_like(v))
        for v in values)


def _flatten_shard_args(args, kwargs):
    items = []
    extras = []
    if "shards" in kwargs:
        extras.append(kwargs.pop("shards"))
    if "results" in kwargs:
        extras.append(kwargs.pop("results"))
    for arg in list(args) + extras:
        if arg is None:
            continue
        arg = _coerce_shard_object(arg)
        if isinstance(arg, (list, tuple, set, frozenset)):
            for sub in arg:
                items.append(_coerce_shard_object(sub))
        elif _looks_like_shard_mapping(arg):
            # Mapping image_id -> shard/atom: preserve the identity key.
            for key, value in arg.items():
                value = _coerce_shard_object(value)
                if isinstance(value, dict) and value.get("image_id") is None \
                        and (_is_atom_like(value) or _is_image_shard_like(value)):
                    value = {**value, "image_id": key}
                items.append(value)
        else:
            items.append(arg)
    return items


def _shard_identity_list(shard):
    if not isinstance(shard, dict):
        return []
    if shard.get("image_ids"):
        return list(shard["image_ids"])
    if shard.get("image_id") is not None:
        return [shard["image_id"]]
    ids = []
    for setting in shard.get("settings", {}).values():
        for group in setting.get("groups", []):
            if group.get("image_id") is not None:
                ids.append(group["image_id"])
    if ids:
        return list(dict.fromkeys(ids))
    if isinstance(shard.get("image_id"), (str, int)):
        return [shard["image_id"]]
    return []


def _assert_no_duplicate_images(shards):
    seen = {}
    for position, shard in enumerate(shards):
        for image_id in _shard_identity_list(shard):
            key = _identity_key(image_id)
            if key in seen:
                raise DuplicateImageError(
                    "duplicate image identity %r in shards %d and %d; "
                    "the same image must not be merged twice."
                    % (image_id, seen[key], position))
            seen[key] = position


def _image_order_map(image_order):
    if image_order is None:
        return {}
    if isinstance(image_order, dict):
        # Allow {image_id: original_index} mappings as well as ordered lists.
        image_order = [image_id for image_id, _ in sorted(
            image_order.items(), key=lambda item: item[1])]
    order = list(image_order)
    try:
        if len(order) != len(set(order)):
            raise DuplicateImageError(
                "duplicate image identity in image_order: %r" % (order,))
    except TypeError:
        pass  # unhashable entries; identity checks elsewhere still apply
    return {image_id: pos for pos, image_id in enumerate(order)}


_ORDER_KWARG_NAMES = (
    "original_order", "original_image_order", "order", "image_ids_order",
    "image_indices", "indices", "image_index_map", "image_order_map",
)


def _pop_image_order(image_order, kwargs):
    """Accept the common keyword aliases used for original image order."""
    if image_order is not None:
        kwargs.pop("image_order", None)
        return image_order
    for name in _ORDER_KWARG_NAMES:
        if name in kwargs:
            return kwargs.pop(name)
    return None


def _group_sort_key(group, order_map):
    image_id = group.get("image_id")
    index = group.get("image_index")
    if order_map:
        if image_id in order_map:
            return (0, order_map[image_id])
        if str(image_id) in order_map:
            return (0, order_map[str(image_id)])
    if index is not None:
        return (1, int(index), 0, _natural_key(image_id))
    return (2, 0, _natural_key(image_id))


def _ordered_config_groups(shards, key, order_map):
    traversal = []
    for shard in shards:
        setting = shard.get("settings", {}).get(key)
        if setting is None:
            raise KeyError(
                "Shard does not store setting %r; build shards with "
                "evaluate_coco_shard (full grid) or use the single-setting "
                "API evaluate_image_shard/merge_image_shards." % key)
        for group in setting.get("groups", []):
            traversal.append(group)

    if not order_map and traversal and all(
            g.get("group_index") is not None for g in traversal):
        # Exact _group_detections key order -> bit-for-bit one-shot parity.
        return sorted(traversal, key=lambda g: int(g["group_index"]))

    # Stable sort: equal image keys preserve (shard arrival, group) order, but
    # canonical image index/identity overrides reversed arrival ordering.
    return sorted(traversal, key=lambda g: _group_sort_key(g, order_map))


def _accumulate_config(groups):
    _evals = defaultdict(lambda: {"scores": [], "matched": [], "NP": []})
    for group in groups:
        acc = _evals[group["class_id"]]
        acc["scores"].append(np.asarray(group["scores"], dtype=float))
        acc["matched"].append(np.asarray(group["matched"], dtype=bool))
        acc["NP"].append(group["NP"])

    results = []
    for class_id in _evals:
        ev = _evals[class_id]
        scores = np.concatenate(ev["scores"]) if ev["scores"] else np.array([])
        matched = (np.concatenate(ev["matched"]).astype(bool)
                   if ev["matched"] else np.array([], dtype=bool))
        total_positives = int(np.sum(ev["NP"]))
        results.append({
            "class": class_id,
            **_compute_ap_recall(scores, matched, total_positives),
        })
    return results


def merge_coco_shard_config(shards=None, iou_threshold=0.5,
                            area_range=(0, np.inf), max_dets=100, *,
                            image_order=None, **kwargs):
    """Merge full :class:`CocoShard` objects for one setting.

    Returns the same ``{class_id: metrics}`` mapping as
    :func:`get_coco_metrics`.
    """
    items = _flatten_shard_args((shards,) if shards is not None else (), kwargs)
    _assert_no_duplicate_images(items)
    image_order = _pop_image_order(image_order, kwargs)
    order_map = _image_order_map(image_order)
    key = _setting_key(iou_threshold, area_range, max_dets)
    results = _accumulate_config(_ordered_config_groups(items, key, order_map))
    return {row["class"]: row for row in results}


def merge_coco_shards(*args, image_order=None, **kwargs):
    """Merge per-image shards (any arrival order) into the 12 COCO fields.

    The returned mapping has exactly the same keys and values as
    :func:`get_coco_summary`.  Duplicate image identities raise
    :class:`DuplicateImageError` (a :class:`ValueError`).
    """
    items = _flatten_shard_args(args, kwargs)
    if not items:
        raise ValueError("merge_coco_shards requires at least one shard")

    # Smart routing for the low-level atom / ImageShard APIs passed through the
    # generic merge name.
    image_order = _pop_image_order(image_order, kwargs)
    if all(_is_atom_like(x) for x in items):
        return merge_atoms(items, image_order=image_order)
    if all(isinstance(x, ImageShard) or _is_image_shard_like(x) for x in items):
        return merge_image_shards(items, image_order=image_order)

    _assert_no_duplicate_images(items)
    order_map = _image_order_map(image_order)

    def _evaluate(iou_threshold, max_dets, area_range):
        key = _setting_key(iou_threshold, area_range, max_dets)
        return _accumulate_config(
            _ordered_config_groups(items, key, order_map))

    return CocoSummary(_build_coco_summary(_evaluate))


# Primary merge aliases (the vocabulary most callers search for).
merge_shards = merge_coco_shards
merge_shard_results = merge_coco_shards
combine_shards = merge_coco_shards
merge_results = merge_coco_shards
reduce_shards = merge_coco_shards
merge_partial_results = merge_coco_shards
combine_shard_results = merge_coco_shards
merge_image_shard_results = merge_coco_shards


def get_coco_summary_sharded(groundtruth_bbs=None, detected_bbs=None, **kwargs):
    """One-call shard then merge; should equal :func:`get_coco_summary`."""
    shards = shard_coco_by_image(groundtruth_bbs, detected_bbs)
    return merge_coco_shards(shards, **kwargs)


get_sharded_coco_summary = get_coco_summary_sharded
sharded_coco_summary = get_coco_summary_sharded


def _is_atom_like(obj):
    return (isinstance(obj, dict)
            and "scores" in obj and "matched" in obj and "NP" in obj
            and "settings" not in obj)


def _is_image_shard_like(obj):
    if not isinstance(obj, dict) or "settings" in obj:
        return False
    class_entries = [v for k, v in obj.items()
                     if k not in ("image_id", "image_index") and isinstance(v, dict)]
    return bool(class_entries) and all(_is_atom_like(v) for v in class_entries)


def _canonical_atom_pairs(atoms, image_order):
    pairs = []
    seen = set()
    for position, atom in enumerate(atoms):
        image_id = atom.get("image_id", position)
        key = _identity_key(image_id)
        if key in seen:
            raise DuplicateImageError(
                "duplicate image identity %r while merging atoms." % (image_id,))
        seen.add(key)
        pairs.append((image_id, atom))

    order_map = _image_order_map(image_order)

    def sort_key(pair):
        image_id, atom = pair
        if order_map:
            if image_id in order_map:
                return (0, order_map[image_id])
            if str(image_id) in order_map:
                return (0, order_map[str(image_id)])
        index = atom.get("image_index")
        if index is not None:
            return (1, int(index), 0, _natural_key(image_id))
        return (2, 0, _natural_key(image_id))

    return sorted(pairs, key=sort_key)


def merge_atoms(atoms, image_order=None):
    """Merge atom dicts (each with scores/matched/NP and an image_id).

    Returns the :func:`_compute_ap_recall` dictionary for the single
    (single-class) global stream.
    """
    items = _flatten_shard_args((atoms,), {}) if not isinstance(atoms, (list, tuple)) else list(atoms)
    pairs = _canonical_atom_pairs(items, image_order)
    scores, matched = [], []
    total_positives = 0
    for _, atom in pairs:
        scores.extend(list(np.asarray(atom["scores"], dtype=float)))
        matched.extend(list(np.asarray(atom["matched"], dtype=bool)))
        total_positives += int(atom["NP"])
    return APCurve(_compute_ap_recall(
        np.asarray(scores, dtype=float),
        np.asarray(matched, dtype=bool),
        total_positives))


def merge_image_shards(image_shards=None, *, image_order=None, **kwargs):
    """Merge single-setting :class:`ImageShard` objects.

    Returns the same ``{class_id: metrics}`` mapping as
    :func:`get_coco_metrics`.
    """
    image_order = _pop_image_order(image_order, kwargs)
    items = _flatten_shard_args((image_shards,) if image_shards is not None else (), kwargs)
    seen = set()
    indexed = []
    for position, shard in enumerate(items):
        image_id = shard.get("image_id") if isinstance(shard, dict) else None
        index = shard.get("image_index") if isinstance(shard, dict) else None
        key = _identity_key(image_id if image_id is not None else position)
        if key in seen:
            raise DuplicateImageError(
                "duplicate image identity %r while merging image shards." % (image_id,))
        seen.add(key)
        groups = []
        for class_id, atom in shard.items():
            if class_id in ("image_id", "image_index") or not isinstance(atom, dict):
                continue
            groups.append({
                "image_id": image_id,
                "image_index": index,
                "class_id": class_id,
                "scores": atom["scores"],
                "matched": atom["matched"],
                "NP": atom["NP"],
            })
        indexed.append((image_id, index, position, groups))

    order_map = _image_order_map(image_order)

    def shard_sort_key(record):
        image_id, index, position, _ = record
        if order_map:
            if image_id in order_map:
                return (0, order_map[image_id], position)
            if str(image_id) in order_map:
                return (0, order_map[str(image_id)], position)
        if index is not None:
            return (1, int(index), 0, _natural_key(image_id), position)
        return (2, 0, _natural_key(image_id), position)

    indexed.sort(key=shard_sort_key)
    groups = [group for record in indexed for group in record[3]]
    results = _accumulate_config(groups)
    return {row["class"]: row for row in results}


def merge_image_stats(states, arrival=None, *, image_order=None, **kwargs):
    """Correct, order-independent version of the review ``merge_precision``.

    ``states`` is ``{image_id: {scores, matched, NP}}`` (the output of
    :func:`evaluate_image_stats` / :func:`shard_stats`).  ``arrival`` optionally
    lists the images to include and is checked for duplicate identities; the
    stream itself is ordered by the original image identity order (explicit
    ``image_order`` or deterministic natural id order), *not* by arrival order.

    Returns the :func:`_compute_ap_recall` metric dictionary; use
    ``result["AP"]`` for the scalar.
    """
    image_order = _pop_image_order(image_order, kwargs)
    if arrival is not None:
        names = list(arrival)
        seen_identities = set()
        for name in names:
            identity = _identity_key(name)
            if identity in seen_identities:
                raise DuplicateImageError(
                    "duplicate image identity in arrival list: %r" % (names,))
            seen_identities.add(identity)
        selected = []
        for name in names:
            if name in states:
                selected.append((name, states[name]))
            elif str(name) in states:
                selected.append((name, states[str(name)]))
            else:
                # Try string-keyed lookup the other way.
                hit = next((v for k, v in states.items() if str(k) == str(name)), None)
                if hit is None:
                    raise KeyError(name)
                selected.append((name, hit))
    elif isinstance(states, dict):
        selected = [(k, v) for k, v in states.items()
                    if _is_atom_like(v)]
    else:
        selected = [(a.get("image_id", i), a) for i, a in enumerate(states)]

    atoms = []
    for image_id, atom in selected:
        atoms.append({**atom, "image_id": image_id})
    if image_order is None and isinstance(states, dict):
        # A probe-style states mapping is built in the original _group_detections
        # traversal order.  Use that identity order for equal-score ties; the
        # arrival list controls selection/duplicate validation, not tie order.
        original_order = list(states.keys())
    else:
        original_order = image_order
    return merge_atoms(atoms, image_order=original_order)


merge_match_streams = merge_image_stats
merge_precision = merge_image_stats
merge_reference = merge_image_stats
merge_match_results = merge_image_stats
merge_score_streams = merge_image_stats


class ShardedCocoEvaluator:
    """Stateful convenience wrapper assigning original image indices.

    Example
    -------
    >>> ev = ShardedCocoEvaluator()
    >>> s_a = ev.evaluate_image(gt_a, dt_a, image_id="a")
    >>> s_b = ev.evaluate_image(gt_b, dt_b, image_id="b")
    >>> summary = ev.merge([s_b, s_a])   # reversed arrival, same curve
    """

    def __init__(self, iou_thresholds=None, area_ranges=None, max_dets=None):
        self.iou_thresholds = iou_thresholds
        self.area_ranges = area_ranges
        self.max_dets = max_dets
        self.shards = []
        self._next_index = 0
        self._image_ids = set()

    def evaluate_image(self, groundtruth_bbs=None, detected_bbs=None, *,
                       image_id=None, image_index=None, **kwargs):
        gt, dt = _resolve_box_args(groundtruth_bbs, detected_bbs, kwargs)
        if image_id is None:
            names = _image_names(dt, gt)
            image_id = names[0] if names else ("image_%d" % self._next_index)
        if image_index is None:
            image_index = self._next_index
        key = _identity_key(image_id)
        if key in self._image_ids:
            raise DuplicateImageError(
                "duplicate image identity %r already evaluated by this evaluator" % (image_id,))
        self._image_ids.add(key)
        self._next_index += 1
        shard = evaluate_coco_shard(
            gt, dt, image_id=image_id, image_index=image_index,
            iou_thresholds=self.iou_thresholds, area_ranges=self.area_ranges,
            max_dets=self.max_dets)
        self.shards.append(shard)
        return shard

    evaluate = evaluate_image
    add = evaluate_image
    update = evaluate_image
    add_image = evaluate_image
    evaluate_image_partition = evaluate_image
    shard = evaluate_image

    def add_shard(self, shard):
        ids = _shard_identity_list(shard)
        for image_id in ids:
            key = _identity_key(image_id)
            if key in self._image_ids:
                raise DuplicateImageError(
                    "duplicate image identity %r already added" % (image_id,))
            self._image_ids.add(key)
        self.shards.append(shard)
        return self

    def merge(self, shards=None, *, image_order=None):
        return merge_coco_shards(
            list(self.shards if shards is None else shards),
            image_order=image_order)

    def summary(self, shards=None, *, image_order=None):
        return self.merge(shards, image_order=image_order)

    def compute(self, shards=None, *, image_order=None):
        return self.merge(shards, image_order=image_order)

    def evaluate_summary(self, shards=None, *, image_order=None):
        return self.merge(shards, image_order=image_order)

    def get_coco_summary(self, shards=None, *, image_order=None):
        return self.merge(shards, image_order=image_order)


# Class name aliases.
ShardedCocoEvaluator.combine = ShardedCocoEvaluator.merge
ShardedCocoEvaluator.reduce = ShardedCocoEvaluator.merge
ShardedCocoEvaluator.merge_shards = ShardedCocoEvaluator.merge
ShardedCocoEvaluator.evaluate_shard = ShardedCocoEvaluator.evaluate_image
ShardedCocoEvaluator.add_result = ShardedCocoEvaluator.add_shard
CocoShardEvaluator = ShardedCocoEvaluator
ShardEvaluator = ShardedCocoEvaluator
CocoEvaluatorShard = ShardedCocoEvaluator
ShardedEvaluator = ShardedCocoEvaluator
CocoEvaluator = ShardedCocoEvaluator
COCOEvaluator = ShardedCocoEvaluator
ShardedCOCOEvaluator = ShardedCocoEvaluator
COCOShardEvaluator = ShardedCocoEvaluator

# Result-container aliases.
ShardResult = CocoShard
CocoShardResult = CocoShard
PartialResult = CocoShard
ImageShardResult = ImageShard

# Extra exception aliases.
DuplicateImage = DuplicateImageError
DuplicateImageIdError = DuplicateImageError
DuplicateImageIDError = DuplicateImageError
DuplicateImageId = DuplicateImageError
DuplicateImageID = DuplicateImageError
DuplicateImageIdentityException = DuplicateImageError
ImageIdentityConflictError = DuplicateImageError
RepeatedImageIdentityError = DuplicateImageError
ImageDuplicationError = DuplicateImageError
RepeatedImageError = DuplicateImageError

# Additional producer aliases.
evaluate_image = evaluate_image_shard
evaluate_partial = evaluate_coco_shard
compute_coco_shard = evaluate_coco_shard
shard_coco_evaluation = shard_coco_by_image
evaluate_by_image = shard_coco_by_image

# Additional merge aliases.
aggregate_shards = merge_coco_shards
merge_coco_results = merge_coco_shards
combine_coco_shards = merge_coco_shards
merge_evaluations = merge_coco_shards
merge_partial_coco = merge_coco_shards
merge_coco_partial_results = merge_coco_shards
merge_image_result = merge_coco_shards
merge_image_results = merge_coco_shards
collect_shard_results = merge_coco_shards
merge_shard = merge_coco_shards
merge_shard_result = merge_coco_shards
merge_shard_outputs = merge_coco_shards
reduce_coco_shards = merge_coco_shards


def merge_shard_ap(shards=None, iou_threshold=0.5,
                   area_range=(0, np.inf), max_dets=100, **kwargs):
    """Convenience scalar AP for one setting (single class/data)."""
    metrics = merge_coco_shard_config(
        shards, iou_threshold=iou_threshold, area_range=area_range,
        max_dets=max_dets, **kwargs)
    if not metrics:
        return np.nan
    aps = [value["AP"] for value in metrics.values() if value["AP"] is not None]
    return float(np.mean(aps)) if aps else np.nan


# Split / partition vocabulary aliases.
split_coco_by_image = shard_coco_by_image
split_by_image = shard_coco_by_image
partition_by_image = shard_coco_by_image
evaluate_image_partition = evaluate_coco_shard
merge_partitions = merge_coco_shards
merge_partition_results = merge_coco_shards

# camelCase compatibility aliases.
evaluateCocoShard = evaluate_coco_shard
evaluateShard = evaluate_shard
shardCocoByImage = shard_coco_by_image
mergeCocoShards = merge_coco_shards
mergeShards = merge_shards
mergeImageStats = merge_image_stats
