import os

import numpy as np
from src.evaluators.pascal_voc_evaluator import (calculate_ap_11_point_interp,
                                                 calculate_ap_every_point)
from src.tube import Tube
from src.utils.enumerators import (MethodAveragePrecision,
                                   TubeMatchingMethod)
from src.utils.read_files import File


class TubeEvaluator():
    """ Spatio-temporal tube evaluator
    """
    def __init__(self,
                 anno_filepath,
                 preds_filepath,
                 method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
                 matching_method=TubeMatchingMethod.MAX_IOU):
        """Class constructor

        Args:
            anno_filepath (str): annotation filepath
            preds_filepath (str): prediction filepath in json extension
            method (MethodAveragePrecision, optional): Recall interpolation method (see src.utils.enumerators). Defaults to MethodAveragePrecision.EVERY_POINT_INTERPOLATION.
            matching_method (TubeMatchingMethod, optional): Strategy used to associate predicted and ground-truth tubes.
                ``TubeMatchingMethod.MAX_IOU`` greedily pairs the tubes with the highest IoU first (legacy behaviour).
                ``TubeMatchingMethod.CONFIDENCE`` processes predictions in decreasing confidence order so that the
                association order matches the order used to build the PR curve. Defaults to TubeMatchingMethod.MAX_IOU.
        """

        if not anno_filepath.endswith('.json'):
            raise ValueError("Invalid extension file: ", anno_filepath)
        if not preds_filepath.endswith('.json'):
            raise ValueError("Invalid extension file:", preds_filepath)

        self._anno_filepath = anno_filepath
        self._preds_filepath = preds_filepath
        self._method = method
        self._matching_method = matching_method

    def __reset(self):
        """reset object
        """
        self._gt = []
        self._predictions = []
        self.nb_tp = 0
        self.nb_fp = 0
        self.nb_fn = 0
        self.nb_tn = -1
        self._res = dict()
        # actual pred<->gt pairings, grouped by class name
        self._pairings = dict()
        # self._video_id = 0

    def __process(self):
        """process the files and prepare to evaluate
        """
        self.__reset()

        annot_data = File(self._anno_filepath).read()
        pred_data = File(self._preds_filepath).read()

        # List with all gts tubes (Ex: [Tube(num_instances=5...99 0.98]]), Tube(num_instances=5...99 0.89]])] )
        self._gt = [Tube(**annot) for annot in annot_data['annotations']]
        # List with all preds tubes
        self._predictions = [Tube(**pred) for pred in pred_data]

        self._videos = annot_data['videos']
        self._classes = annot_data['categories']

    def evaluate(self, thr=0.5, matching_method=None):
        """Evaluate the predictions according to the chosen IOU threshold

        Args:
            thr (float, optional): IOU threshold 0 < thr < 1. Defaults to 0.5.
            matching_method (TubeMatchingMethod, optional): Strategy used to associate predicted and
                ground-truth tubes. When None, the matching method given to the constructor is used.
                Defaults to None.

        Returns:
            res, mAP: return a dictionary (res) with results per class. Also, returns the mAP.
        """
        if not 0 < thr <= 1:
            raise ValueError("IOU threshold must be 0 < thr <= 1: ", thr)

        if matching_method is None:
            matching_method = self._matching_method
        if not isinstance(matching_method, TubeMatchingMethod):
            raise ValueError(f'Invalid tube matching method: {matching_method}')

        self.__process()

        # loop over classes
        # TODO: group detection on videos and classes basis to avoid loop multiple times
        for obj_cls in self._classes:
            gt_cls = [gt for gt in self._gt if gt.category_id == obj_cls['id']]
            preds_cls = [pred for pred in self._predictions if pred.category_id == obj_cls['id']]

            # sort detections by decreasing confidence. Ties keep the input order, so
            # predictions with the same confidence are always processed in a reproducible way.
            preds_cls = sorted(enumerate(preds_cls),
                               key=lambda enum_tube: enum_tube[1].confidence,
                               reverse=True)
            preds_cls = [pred for _, pred in preds_cls]

            class_pairings = []
            n_tp = n_fp = n_fn = 0

            # loop over videos
            for vid_id in self._videos:
                gts = [gt for gt in gt_cls if gt.video_id == vid_id['id']]
                preds = [pred for pred in preds_cls if pred.video_id == vid_id['id']]

                v_tp, v_fp, v_fn, v_pairings = self._classify_tubes(
                    preds, gts, thr, matching_method)
                n_tp += v_tp
                n_fp += v_fp
                n_fn += v_fn
                class_pairings.extend(v_pairings)

            TP = np.array([int(tube.isTP) for tube in preds_cls])
            FP = np.logical_not(TP).astype(int)

            # compute precision, recall and average precision
            acc_TP = np.cumsum(TP)
            acc_FP = np.cumsum(FP)
            rec = acc_TP / len(gt_cls)
            prec = np.divide(acc_TP, (acc_FP + acc_TP))

            # Depending on the method, call the right implementation
            if self._method == MethodAveragePrecision.EVERY_POINT_INTERPOLATION:
                [ap, mpre, mrec, ii] = calculate_ap_every_point(rec, prec)
            elif self._method == MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION:
                [ap, mpre, mrec, _] = calculate_ap_11_point_interp(rec, prec)
            else:
                raise ValueError(f'Invalid interpolation method: {self._method}')

            # store the actual pred<->gt pairings so they can be inspected later
            self._pairings[obj_cls['name']] = class_pairings

            # add class result in the dictionary to be returned
            self._res[obj_cls['name']] = {
                'precision': prec,
                'recall': rec,
                'AP': ap,
                'interpolated precision': mpre,
                'interpolated recall': mrec,
                'total TP': n_tp,
                'total FP': n_fp,
                'total FN': n_fn,
                'matching method': matching_method,
                'pairings': class_pairings,
            }
        # For mAP, only the classes in the gt set should be considered
        mAP = 0.0
        for c, r in self._res.items():
            if any(cat['name'] == c for cat in self._classes):
                mAP += r['AP']
        mAP /= len(self._classes)

        return self._res, mAP

    def get_pairings(self):
        """Return the actual pred<->gt pairings produced by the last evaluation.

        The result is a dictionary indexed by class name. Each value is a list with one
        entry per prediction containing:
            - 'pred_id': id of the predicted tube (its position in the predictions when no id is set);
            - 'gt_id': id of the matched ground-truth tube (None when the prediction is a FP);
            - 'iou': spatio-temporal IoU with the matched (or best overlapping) ground-truth tube;
            - 'status': 'TP', 'duplicate FP' or 'FP';
            - 'confidence': confidence of the predicted tube;
            - 'video_id': video the predicted tube belongs to.

        Returns:
            dict: pairings grouped by class name.
        """
        if not hasattr(self, '_pairings'):
            return dict()
        return self._pairings

    def _classify_tubes(self, preds: list, gts: list, thr: float,
                        matching_method: TubeMatchingMethod) -> tuple:
        """Classify the `preds` in TP or !TP and the `gts` in FN or !FN, by setting an attribute
        in each tube in the lists. This is done according to the threshold and the matching strategy.

        Args:
            preds (list): list of  predicted Tube objects
            gts (list): list of  annotation Tube objects
            thr (float): threshold to consider a tube correctly detected. It compares the Spatio-temporal IOU (STT-IOU).
            matching_method (TubeMatchingMethod): strategy used to associate predictions and ground truths.

        Returns:
            tuple: return the number of TP, FP, FN and the list of pred<->gt pairings.
        """
        if matching_method == TubeMatchingMethod.CONFIDENCE:
            return self._classify_tubes_confidence_first(preds, gts, thr)
        return self._classify_tubes_max_iou(preds, gts, thr)

    def _classify_tubes_confidence_first(self, preds: list, gts: list,
                                         thr: float) -> tuple:
        """Confidence-first association (PASCAL VOC style).

        Predictions are processed in decreasing confidence order - the same order used to build
        the PR curve - and each prediction is used exactly once. For every prediction, the
        still-available ground-truth tube with the highest IoU is selected. A prediction is a TP
        when this IoU reaches the threshold and the ground truth has not been taken yet, otherwise
        it is a FP (a 'duplicate FP' when it overlaps an already matched ground truth above the
        threshold). Ties in confidence keep the input order, making the result reproducible.

        Args:
            preds (list): list of predicted Tube objects, already sorted by decreasing confidence
            gts (list): list of annotation Tube objects
            thr (float): threshold to consider a tube correctly detected

        Returns:
            tuple: number of TP, FP, FN and the list of pred<->gt pairings
        """
        overlaps = self._tube_pairwise_iou(preds, gts)

        # consider no detections at first
        [gt.__setattr__('isFN', True) for gt in gts]
        [pred.__setattr__('isTP', False) for pred in preds]
        [pred.__setattr__('matched_gt_id', None) for pred in preds]
        [pred.__setattr__('matched_iou', None) for pred in preds]
        [pred.__setattr__('match_status', None) for pred in preds]

        gt_taken = np.zeros(len(gts), dtype=bool)
        pairings = []
        nb_tp = 0

        for pred_idx, pred in enumerate(preds):
            best_gt_idx = None
            best_iou = None

            if len(gts) > 0:
                # each prediction is used exactly once: pick the best *available* gt tube
                available = np.where(np.logical_not(gt_taken), overlaps[pred_idx, :], -1.0)
                best_gt_idx = int(np.argmax(available))
                best_iou = float(available[best_gt_idx])

            if best_iou is not None and best_iou >= thr:
                # first prediction reaching this ground truth -> TP
                nb_tp += 1
                matched_gt_id = self._tube_id(gts[best_gt_idx], best_gt_idx)
                pred.__setattr__('isTP', True)
                pred.__setattr__('matched_gt_id', matched_gt_id)
                pred.__setattr__('matched_iou', best_iou)
                pred.__setattr__('match_status', 'TP')
                gts[best_gt_idx].__setattr__('isFN', False)
                gt_taken[best_gt_idx] = True
                status = 'TP'
            else:
                # either the overlap is below the threshold, or every gt is already matched
                pred.__setattr__('isTP', False)
                if len(gts) > 0 and float(overlaps[pred_idx].max()) >= thr:
                    status = 'duplicate FP'
                else:
                    status = 'FP'
                pred.__setattr__('match_status', status)
                matched_gt_id = None
                # report the best raw overlap even when no ground truth can be matched
                if len(gts) > 0:
                    raw_best = int(np.argmax(overlaps[pred_idx]))
                    best_gt_idx = raw_best
                    best_iou = float(overlaps[pred_idx, raw_best])
                    pred.__setattr__('matched_iou', best_iou)

            pairings.append({
                'pred_id': self._tube_id(pred, pred_idx),
                'gt_id': matched_gt_id,
                'iou': best_iou,
                'status': status,
                'confidence': None if pred.confidence is None else float(pred.confidence),
                'video_id': pred.video_id,
            })

        nb_fp = len(preds) - nb_tp
        nb_fn = len(gts) - nb_tp
        assert nb_fp >= 0
        assert nb_fn >= 0

        return nb_tp, nb_fp, nb_fn, pairings

    def _classify_tubes_max_iou(self, preds: list, gts: list, thr: float) -> tuple:
        """Legacy maximum-overlap-first association.

        The pred<->gt pair with the highest IoU over the whole video is built first, then the
        involved tubes are removed and the process repeats. Tubes with the highest overlaps
        therefore have priority, regardless of their confidence.

        Args:
            preds (list): list of  predicted Tube objects
            gts (list): list of  annotation Tube objects
            thr (float): threshold to consider a tube correctly detected. It compares the Spatio-temporal IOU (STT-IOU).

        Returns:
            tuple: number of TP, FP, FN and the list of pred<->gt pairings
        """

        gt_overlaps = np.zeros(len(gts))
        overlaps = self._tube_pairwise_iou(preds, gts)

        # consider no detections at first
        [gt.__setattr__('isFN', True) for gt in gts]
        [pred.__setattr__('isTP', False) for pred in preds]
        [pred.__setattr__('matched_gt_id', None) for pred in preds]
        [pred.__setattr__('matched_iou', None) for pred in preds]
        [pred.__setattr__('match_status', None) for pred in preds]

        # pred_idx -> (gt_idx, iou) for every pair built by the greedy loop
        greedy_pairs = dict()

        for j in range(min(len(preds), len(gts))):
            max_overlaps = overlaps.max(axis=0)
            argmax_overlaps = overlaps.argmax(axis=0)

            # find which gt tube is 'best' covered (i.e. 'best' = most iou)
            gt_ovr = max_overlaps.max(axis=0)
            gt_ind = max_overlaps.argmax(axis=0)
            assert gt_ovr >= 0

            # find the proposal tube that covers the best covered gt tube
            box_ind = argmax_overlaps[gt_ind]

            # in case the ovr greater than threshold,
            # set the gt tube to detected and pred as TP
            if gt_ovr >= thr:
                gts[gt_ind].__setattr__('isFN', False)
                preds[box_ind].__setattr__('isTP', True)
                preds[box_ind].__setattr__('matched_gt_id', self._tube_id(gts[gt_ind], gt_ind))
                preds[box_ind].__setattr__('match_status', 'TP')
            else:
                preds[box_ind].__setattr__('match_status', 'FP')

            preds[box_ind].__setattr__('matched_iou', float(gt_ovr))

            # record the iou coverage of this gt tube
            gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert gt_overlaps[j] == gt_ovr
            greedy_pairs[box_ind] = (gt_ind, float(gt_ovr))

            # mark the proposal tube and the gt tube as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        nb_tp = (gt_overlaps >= thr).astype(int).sum()
        assert nb_tp >= 0

        nb_fp = len(preds) - nb_tp
        assert nb_fp >= 0

        nb_fn = len(gts) - nb_tp
        assert nb_fn >= 0

        # expose the actual pairings: one entry per prediction, in the input order
        raw_overlaps = self._tube_pairwise_iou(preds, gts)
        pairings = []
        for pred_idx, pred in enumerate(preds):
            if pred_idx in greedy_pairs:
                gt_ind, iou = greedy_pairs[pred_idx]
                if iou >= thr:
                    status = 'TP'
                    matched_gt_id = self._tube_id(gts[gt_ind], gt_ind)
                else:
                    status = 'FP'
                    matched_gt_id = None
            else:
                # prediction never selected by the greedy loop: report its best raw overlap
                status = 'FP'
                matched_gt_id = None
                if len(gts) > 0:
                    gt_ind = int(np.argmax(raw_overlaps[pred_idx]))
                    iou = float(raw_overlaps[pred_idx, gt_ind])
                    if iou >= thr:
                        # it overlaps a gt above the threshold but that gt was already taken
                        status = 'duplicate FP'
                else:
                    iou = None
            pred.__setattr__('match_status', status)
            pairings.append({
                'pred_id': self._tube_id(pred, pred_idx),
                'gt_id': matched_gt_id,
                'iou': iou,
                'status': status,
                'confidence': None if pred.confidence is None else float(pred.confidence),
                'video_id': pred.video_id,
            })

        return nb_tp, nb_fp, nb_fn, pairings

    @staticmethod
    def _tube_id(tube: Tube, fallback_idx: int):
        """Return the tube 'id' when available, otherwise its position in the list.

        Args:
            tube (Tube): tube object
            fallback_idx (int): position of the tube in its list when no id attribute exists

        Returns:
            int: identifier of the tube
        """
        return getattr(tube, 'id', fallback_idx)

    def _tube_pairwise_iou(self, preds: list, gts: list) -> np.array:
        """compute the pairwise spatio-temporal tube iou (STT-IOU)

        Args:
            preds (list): list of tube predictions
            gts (list): list of tube annotation

        Returns:
            np.array: pairwise STT-IOU
        """
        # initialize matrices that will keep the intersection and union of the tubes
        inter = np.zeros((len(preds), len(gts)))
        union = np.zeros((len(preds), len(gts)))

        pred_idx = 0
        for pred in preds:
            gt_idx = 0

            for gt in gts:
                # compute the pairwise intersection
                inter[pred_idx, gt_idx] = self._tubes_inter(pred, gt)
                # compute the pairwise union
                union[pred_idx, gt_idx] = self._tubes_union(pred, gt)

                gt_idx += 1

            pred_idx += 1

        iou = inter / (union - inter)
        return iou

    def _tubes_inter(self, tube1: Tube, tube2: Tube) -> np.array:
        """Give two tubes of track size N and M,
        compute de intersection volume between __all__ N x M pairs of boxes.
        The boxes in track must be (xmin, ymin, xmax, ymax)

        Args:
            tube1, tube2 (Tube): two `Tubes`. Contains tracks with N & M boxes, respectively.

        Returns:
            np.array: intersection volume.
        """

        inter_frames = self._get_intersection_frames(tube1, tube2)

        inter = 0

        if inter_frames is not None:
            for inter_frame in inter_frames:
                box1 = tube1.get_frame_boxes(inter_frame)
                box2 = tube2.get_frame_boxes(inter_frame)

                width_height = np.minimum(box1[:, None, 2:], box2[:, 2:]) - np.maximum(
                    box1[:, None, :2], box2[:, :2])  # [N,M,2]

                width_height.clip(min=0, out=width_height)  # [N,M,2]
                inter += width_height.prod(axis=2).sum()  # scalar intersection volume
                del width_height

        return inter

    def _tubes_union(self, tube1: Tube, tube2: Tube) -> np.array:
        """Give two tubes of track size N and M,
        compute de union volume between __all__ N x M pairs of boxes.

        Args:
            tube1, tube2 (Tube): two `Tubes`. Contains tracks with N & M boxes, respectively.

        Returns:
            np.array: union volume.
        """
        vol1 = tube1.get_tube_volume()
        vol2 = tube2.get_tube_volume()
        return vol1 + vol2

    def _get_intersection_frames(self, tube1: Tube, tube2: Tube) -> list:
        """Get the frames that are commom to tube1 and tube2.

        Args:
            tube1, tube2 (Tube): two `Tubes`. Contains tracks with N & M boxes, respectively.

        Returns:
            list: list of frames commom to tube1 and tube2.
        """
        frames_tube1 = set(tube1.get_frames())
        frames_tube2 = set(tube2.get_frames())

        frames_inter = frames_tube1.intersection(frames_tube2)

        if len(frames_inter) > 0:
            return frames_inter

        return None
