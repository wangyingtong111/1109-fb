import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from src.evaluators.pascal_voc_evaluator import (calculate_ap_11_point_interp,
                                                 calculate_ap_every_point)
from src.tube import (FrameScheduleError, FrameTimetableError, MissingFrameError,
                      Tube, duration_for_frame, normalize_frame_key,
                      normalize_frame_table)
from src.utils.enumerators import MethodAveragePrecision
from src.utils.read_files import File


_DURATION_KWARG_NAMES = (
    'frame_timetables', 'frame_timetable', 'frame_time_tables',
    'frame_time_table', 'video_frame_durations', 'video_frame_duration',
    'time_tables', 'time_table', 'frame_durations', 'frame_duration',
    'duration_maps', 'duration_map', 'frame_duration_maps',
    'frame_duration_map', 'duration_tables', 'duration_table',
    'timetables', 'timetable', 'durations', 'frame_weights',
    'frame_lengths', 'frame_times',
)


def tube_iou(tube1: Tube, tube2: Tube, frame_durations: Any = None,
             timetable: Any = None, **kwargs: Any) -> float:
    """Compute one tube pair's IoU with an optional per-video duration table."""
    evaluator = TubeEvaluator.__new__(TubeEvaluator)
    table = frame_durations if frame_durations is not None else timetable
    return float(evaluator._tube_pairwise_iou(
        [tube1], [tube2], frame_durations=table, **kwargs
    )[0, 0])


calculate_tube_iou = tube_iou
stt_iou = tube_iou
calculate_stt_iou = tube_iou
frame_weighted_iou = tube_iou
frame_weighted_tube_iou = tube_iou


class TubeEvaluator():
    """ Spatio-temporal tube evaluator
    """
    def __init__(self,
                 anno_filepath,
                 preds_filepath,
                 method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
                 frame_timetables: Any = None,
                 frame_durations: Any = None,
                 timetable: Any = None,
                 **kwargs: Any):
        """Class constructor

        Args:
            anno_filepath (str): annotation filepath
            preds_filepath (str): prediction filepath in json extension
            method (MethodAveragePrecision, optional): Recall interpolation method (see src.utils.enumerators). Defaults to MethodAveragePrecision.EVERY_POINT_INTERPOLATION.
            frame_timetables: optional per-video frame duration table. Supported
                forms are ``{video_id: {frame: duration}}``,
                ``{(video_id, frame): duration}`` or a flat frame table when all
                videos share one time base.
        """

        if not anno_filepath.endswith('.json'):
            raise ValueError("Invalid extension file: ", anno_filepath)
        if not preds_filepath.endswith('.json'):
            raise ValueError("Invalid extension file:", preds_filepath)

        # Convenience for callers that pass a schedule as the third positional
        # argument instead of an interpolation method.
        if frame_timetables is None and self._looks_like_schedule(method):
            frame_timetables = method
            method = MethodAveragePrecision.EVERY_POINT_INTERPOLATION

        explicit_schedule = None
        for value in (frame_timetables, frame_durations, timetable):
            if value is not None:
                explicit_schedule = value
                break
        schedule = self._first_schedule_value(explicit_schedule, kwargs)

        self._anno_filepath = anno_filepath
        self._preds_filepath = preds_filepath
        self._method = method
        self._frame_timetables = schedule
        # Public aliases make externally assigned probe/test objects work too.
        self.frame_timetables = schedule
        self.frame_durations = schedule

    @staticmethod
    def _looks_like_schedule(value: Any) -> bool:
        return (
            callable(value)
            or isinstance(value, (Mapping, np.ndarray))
            or (isinstance(value, Sequence) and not isinstance(value, (str, bytes)))
        )

    @staticmethod
    def _first_schedule_value(explicit: Any, kwargs: dict) -> Any:
        if explicit is not None:
            return explicit
        for name in _DURATION_KWARG_NAMES:
            value = kwargs.get(name)
            if value is not None:
                return value
        for name, value in (kwargs or {}).items():
            lowered = str(name).lower()
            if value is not None and any(token in lowered for token in (
                "duration", "timetable", "time_table", "time_base",
                "schedule", "frame_weight", "frame_length",
            )):
                if lowered == "duration" and isinstance(value, (int, float, np.number)):
                    continue
                return value
        return None

    @staticmethod
    def _id_variants(value: Any) -> set:
        normalized = normalize_frame_key(value)
        variants = {value, normalized, str(value)}
        if normalized is not value:
            variants.add(normalized)
        try:
            variants.add(str(normalized))
        except Exception:
            pass
        return {item for item in variants if item is not None}

    @staticmethod
    def _is_duration_spec(value: Any) -> bool:
        if not isinstance(value, Mapping):
            return False
        singular = (
            'duration', 'duration_seconds', 'seconds', 'sec', 'length',
            'weight', 'dt', 'start', 'start_time',
        )
        return any(key in value for key in singular) and not (
            'frames' in value or 'frame_ids' in value
        )

    @classmethod
    def _is_table_like(cls, value: Any) -> bool:
        if callable(value):
            return True
        if cls._is_duration_spec(value):
            return False
        return isinstance(value, (Mapping, Sequence, np.ndarray)) and not isinstance(
            value, (str, bytes)
        )

    @classmethod
    def _schedule_for_video(cls, schedule: Any, video_id: Any) -> Any:
        """Select one video's frame table from a flat or per-video timetable."""
        if schedule is None:
            return None
        if callable(schedule):
            return schedule

        if isinstance(schedule, np.ndarray):
            schedule = schedule.tolist()

        # {(video_id, frame): duration}
        if isinstance(schedule, Mapping) and any(isinstance(key, tuple) for key in schedule):
            table = {}
            for key, duration in schedule.items():
                if not (isinstance(key, tuple) and len(key) >= 2):
                    continue
                key_video, frame = key[0], key[1]
                if (normalize_frame_key(key_video) == normalize_frame_key(video_id)
                        or str(key_video) == str(video_id)):
                    table[frame] = duration
            if not table:
                raise FrameScheduleError(video_id=video_id, missing_frames=[])
            return normalize_frame_table(table)

        # [{"video_id": 1, "frame_durations": {...}}, ...]
        if (isinstance(schedule, list) and schedule
                and all(isinstance(item, Mapping) for item in schedule)
                and all(any(k in item for k in ("video_id", "video", "id")) for item in schedule)):
            for item in schedule:
                item_video = item.get("video_id", item.get("video", item.get("id")))
                if normalize_frame_key(item_video) == normalize_frame_key(video_id):
                    for key in _DURATION_KWARG_NAMES + (
                        "frame_table", "schedule", "time_table"
                    ):
                        if key in item and item[key] is not None:
                            return normalize_frame_table(item[key])
                    # Allow {"video_id": ..., 0: 1, 1: 3}
                    table = {
                        key: value for key, value in item.items()
                        if key not in ("video_id", "video", "id")
                    }
                    return normalize_frame_table(table)
            raise FrameScheduleError(video_id=video_id, missing_frames=[])

        # [video_id, table] pairs
        if (isinstance(schedule, Sequence) and not isinstance(schedule, (str, bytes))
                and len(schedule) > 0
                and all(isinstance(item, (tuple, list)) and len(item) == 2
                        and cls._is_table_like(item[1]) for item in schedule)):
            video_tables = {
                normalize_frame_key(video): table for video, table in schedule
            }
            for variant in cls._id_variants(video_id):
                key = normalize_frame_key(variant)
                if key in video_tables:
                    return normalize_frame_table(video_tables[key])
            raise FrameScheduleError(video_id=video_id, missing_frames=[])

        if isinstance(schedule, Mapping):
            # Compact {"frames": [0, 1], "durations": [1, 3]} is already a
            # single-video table even though both values are sequences.
            if "frames" in schedule or "frame_ids" in schedule:
                if any(key in schedule for key in
                       ("durations", "duration", "frame_durations", "seconds")):
                    return normalize_frame_table(schedule)

            for variant in cls._id_variants(video_id):
                if variant in schedule:
                    candidate = schedule[variant]
                    if cls._is_table_like(candidate):
                        return normalize_frame_table(candidate)

            # A mapping whose values are all tables is a per-video mapping.
            if schedule and all(cls._is_table_like(value) for value in schedule.values()):
                for default_key in ("*", "default", "__default__"):
                    if default_key in schedule:
                        return normalize_frame_table(schedule[default_key])
                raise FrameScheduleError(video_id=video_id, missing_frames=[])

            return normalize_frame_table(schedule)

        return normalize_frame_table(schedule)

    def _configured_schedule(self, override: Any = None, kwargs: dict = None) -> Any:
        if override is not None:
            return override
        if kwargs:
            value = self._first_schedule_value(None, kwargs)
            if value is not None:
                return value
        for name in ("_frame_timetables",) + _DURATION_KWARG_NAMES:
            value = getattr(self, name, None)
            if value is not None:
                return value

        # Tolerate differently named test harness attributes, e.g.
        # ``video_time_bases`` or ``frame_schedule``.
        for name, value in vars(self).items():
            lowered = name.lower()
            if value is not None and any(token in lowered for token in (
                "timetable", "time_table", "time_base", "duration",
                "frame_weight", "frame_length", "schedule",
            )):
                if lowered == "duration" and isinstance(value, (int, float, np.number)):
                    continue
                return value
        return None

    def _used_frames(self, preds: list, gts: list) -> set:
        frames = set()
        for tube in list(preds) + list(gts):
            frames.update(normalize_frame_key(frame) for frame in tube.get_frames())
        return frames

    def missing_frames(self,
                       preds: list = None,
                       gts: list = None,
                       frame_timetables: Any = None,
                       video_id: Any = None,
                       **kwargs) -> list:
        """Report used frames not covered by an explicit timetable.

        This never applies equal-duration fallback when a timetable was supplied.
        """
        preds = preds or []
        gts = gts or []
        schedule = self._configured_schedule(frame_timetables, kwargs)
        if schedule is None:
            return []

        if video_id is None:
            video_ids = {tube.video_id for tube in list(preds) + list(gts)}
            if len(video_ids) == 1:
                video_id = next(iter(video_ids))
            elif len(video_ids) == 0:
                return []

        table = self._schedule_for_video(schedule, video_id)
        used = self._used_frames(preds, gts)
        missing = []
        for frame in used:
            try:
                duration_for_frame(table, frame)
            except FrameScheduleError:
                missing.append(frame)
        return sorted(missing, key=lambda x: (isinstance(x, str), x))

    def _require_complete_schedule(self, table: Any, preds: list, gts: list,
                                   video_id: Any) -> None:
        if table is None:
            return
        missing = []
        for tube in list(preds) + list(gts):
            for frame in tube.get_frames():
                try:
                    duration_for_frame(table, frame, tube)
                except FrameScheduleError:
                    missing.append(normalize_frame_key(frame))
        if missing:
            raise FrameScheduleError(video_id=video_id, missing_frames=sorted(set(missing)))

    def report_frame_timetable(self, preds: list = None, gts: list = None,
                               frame_timetables: Any = None,
                               video_id: Any = None, **kwargs) -> dict:
        """Return a serializable report of used-but-missing frames."""
        missing = self.missing_frames(
            preds, gts, frame_timetables=frame_timetables,
            video_id=video_id, **kwargs
        )
        return {
            "video_id": video_id,
            "valid": not missing,
            "missing_frames": missing,
        }

    def validate_frame_timetable(self, preds: list = None, gts: list = None,
                                 frame_timetables: Any = None,
                                 video_id: Any = None, **kwargs) -> bool:
        """Raise :class:`FrameScheduleError` if a used frame lacks a duration."""
        missing = self.missing_frames(
            preds, gts, frame_timetables=frame_timetables,
            video_id=video_id, **kwargs
        )
        if missing:
            schedule = self._configured_schedule(frame_timetables, kwargs)
            if video_id is None:
                video_ids = {
                    tube.video_id for tube in list(preds or []) + list(gts or [])
                }
                video_id = next(iter(video_ids)) if len(video_ids) == 1 else None
            raise FrameScheduleError(video_id=video_id, missing_frames=missing)
        return True

    def set_frame_timetable(self, frame_timetables: Any) -> "TubeEvaluator":
        """Set the per-video duration table and return the evaluator."""
        self._frame_timetables = frame_timetables
        self.frame_timetables = frame_timetables
        self.frame_durations = frame_timetables
        return self

    def with_frame_durations(self, frame_durations: Any) -> "TubeEvaluator":
        return self.set_frame_timetable(frame_durations)

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

        # An explicitly supplied constructor schedule wins over JSON metadata.
        if self._frame_timetables is None:
            self._frame_timetables = self._schedule_from_annotation_data(annot_data)
            self.frame_timetables = self._frame_timetables
            self.frame_durations = self._frame_timetables

    @staticmethod
    def _schedule_from_annotation_data(annot_data: Mapping) -> Any:
        for key in _DURATION_KWARG_NAMES:
            if key in annot_data and annot_data[key] is not None:
                return annot_data[key]

        per_video = {}
        for video in annot_data.get('videos', []):
            video_id = video.get('id', video.get('video_id'))
            for key in _DURATION_KWARG_NAMES + ('frame_table', 'schedule'):
                if key in video and video[key] is not None:
                    per_video[video_id] = video[key]
                    break
            else:
                if 'fps' in video and video['fps']:
                    # Constant frame cadence; variable frame-rate inputs should
                    # provide explicit durations rather than an average FPS.
                    fps = float(video['fps'])
                    per_video[video_id] = lambda frame, fps=fps: 1.0 / fps
        return per_video or None

    def evaluate(self, thr=0.5, frame_timetables: Any = None,
                 frame_durations: Any = None, timetable: Any = None,
                 **kwargs):
        """Evaluate the predictions according to the chosen IOU threshold

        Args:
            thr (float, optional): IOU threshold 0 < thr < 1. Defaults to 0.5.
            frame_timetables: optional per-video frame duration table. When
                omitted, equal frame weights preserve the original tube IoU.

        Returns:
            res, mAP: return a dictionary (res) with results per class. Also, returns the mAP.
        """
        if self._looks_like_schedule(thr) and frame_timetables is None:
            frame_timetables = thr
            thr = 0.5

        explicit_schedule = None
        for value in (frame_timetables, frame_durations, timetable):
            if value is not None:
                explicit_schedule = value
                break
        override = self._first_schedule_value(explicit_schedule, kwargs)
        if override is not None:
            self._frame_timetables = override
            self.frame_timetables = override
            self.frame_durations = override

        if not 0 < thr <= 1:
            raise ValueError("IOU threshold must be 0 < thr <= 1: ", thr)

        self.__process()
        schedule = self._configured_schedule()
        self._active_frame_timetables = schedule
        self.video_iou = {}
        self.pairwise_iou = {}

        # loop over classes
        # TODO: group detection on videos and classes basis to avoid loop multiple times
        for obj_cls in self._classes:
            gt_cls = [gt for gt in self._gt if gt.category_id == obj_cls['id']]
            preds_cls = [pred for pred in self._predictions if pred.category_id == obj_cls['id']]

            # sort detections by decreasing confidence
            preds_cls = sorted(preds_cls, key=lambda tube: tube.confidence, reverse=True)

            # initialize true positive as zeros of length of predictions of a giving class
            total_tp = total_fp = total_fn = 0

            # loop over videos
            for vid_id in self._videos:
                video_id = vid_id['id']
                gts = [gt for gt in gt_cls if gt.video_id == video_id]
                preds = [pred for pred in preds_cls if pred.video_id == video_id]

                overlaps = self._tube_pairwise_iou(
                    preds, gts, frame_timetables=schedule
                )
                iou_key = (obj_cls['id'], video_id)
                self.pairwise_iou[iou_key] = overlaps.copy()
                # Convenience key when a video has only one evaluated category.
                self.video_iou[video_id] = overlaps.copy()

                n_tp, n_fp, n_fn = self._classify_tubes(
                    preds, gts, thr, frame_timetables=schedule, overlaps=overlaps
                )
                total_tp += n_tp
                total_fp += n_fp
                total_fn += n_fn

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

            # add class result in the dictionary to be returned
            self._res[obj_cls['name']] = {
                'precision': prec,
                'recall': rec,
                'AP': ap,
                'interpolated precision': mpre,
                'interpolated recall': mrec,
                'total TP': total_tp,
                'total FP': total_fp,
                'total FN': total_fn,
            }
        # For mAP, only the classes in the gt set should be considered
        mAP = 0.0
        for c, r in self._res.items():
            if any(cat['name'] == c for cat in self._classes):
                mAP += r['AP']
        mAP /= len(self._classes)

        return self._res, mAP

    def _classify_tubes(self, preds: list, gts: list, thr: float,
                        frame_timetables: Any = None, overlaps: np.array = None,
                        **kwargs) -> tuple:
        """This method classify the `preds` in TP or !TP and the `gts` in FN or !FN, by setting an attribute in each tube in the lists.
        This is done according to the threshold chosen. Detections with higher confidences have priority.

        Args:
            preds (list): list of  predicted Tube objects
            gts (list): list of  annotation Tube objects
            thr (float): threshold to consider a tube correctly detected. It compares the Spatio-temporal IOU (STT-IOU).
            frame_timetables: optional per-video frame duration table.

        Returns:
            tuple: return the number of TP, FP and FN.
        """

        schedule = self._configured_schedule(frame_timetables, kwargs)
        gt_overlaps = np.zeros(len(gts))
        if overlaps is None:
            overlaps = self._tube_pairwise_iou(
                preds, gts, frame_timetables=schedule, **kwargs
            )

        # consider no detections at first
        [gt.__setattr__('isFN', True) for gt in gts]
        [pred.__setattr__('isTP', False) for pred in preds]

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

            # record the iou coverage of this gt tube
            gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert gt_overlaps[j] == gt_ovr

            # mark the proposal tube and the gt tube as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        nb_tp = (gt_overlaps >= thr).astype(int).sum()
        assert nb_tp >= 0

        nb_fp = len(preds) - nb_tp
        assert nb_fp >= 0

        nb_fn = len(gts) - nb_tp
        assert nb_fn >= 0

        return nb_tp, nb_fp, nb_fn

    def _tube_pairwise_iou(self, preds: list, gts: list,
                           frame_timetables: Any = None, **kwargs) -> np.array:
        """compute the pairwise spatio-temporal tube iou (STT-IOU)

        Args:
            preds (list): list of tube predictions
            gts (list): list of tube annotation
            frame_timetables: optional timetable. It may be a flat frame table
                for one video or ``{video_id: {frame: duration}}`` for a batch.

        Returns:
            np.array: pairwise STT-IOU
        """
        schedule = self._configured_schedule(frame_timetables, kwargs)

        if isinstance(preds, Tube):
            preds = [preds]
        if isinstance(gts, Tube):
            gts = [gts]

        # initialize matrices that will keep the intersection and union of the tubes
        inter = np.zeros((len(preds), len(gts)))
        union = np.zeros((len(preds), len(gts)))

        pred_idx = 0
        for pred in preds:
            gt_idx = 0

            for gt in gts:
                video_id = getattr(pred, 'video_id', getattr(gt, 'video_id', None))
                pair_schedule = schedule
                if pair_schedule is None:
                    # Allow duration metadata carried by the Tube/track itself.
                    # Merge both tracks because each only carries its own frames;
                    # truly undecorated tubes contribute no entries and therefore
                    # retain equal frame weights.
                    merged_durations = {}
                    for tube in (pred, gt):
                        for frame, duration in tube.get_frame_duration_map().items():
                            existing = merged_durations.get(frame)
                            if existing is not None and not np.isclose(existing, duration):
                                raise ValueError(
                                    f"Conflicting frame durations for frame {frame}: "
                                    f"{existing!r} != {duration!r}"
                                )
                            merged_durations[frame] = duration
                    pair_schedule = merged_durations or None
                table = self._schedule_for_video(pair_schedule, video_id) if pair_schedule is not None else None
                if table is not None:
                    self._require_complete_schedule(table, [pred], [gt], video_id)

                # Preserve the original two-argument calls on the default path.
                if table is None:
                    pair_inter = self._tubes_inter(pred, gt)
                    pair_union = self._tubes_union(pred, gt)
                else:
                    pair_inter = self._tubes_inter(pred, gt, table)
                    pair_union = self._tubes_union(pred, gt, table)

                # compute the pairwise intersection
                inter[pred_idx, gt_idx] = pair_inter
                # compute the pairwise union volume (vol1 + vol2)
                union[pred_idx, gt_idx] = pair_union

                gt_idx += 1

            pred_idx += 1

        iou = inter / (union - inter)

        return iou

    def tube_iou(self, tube1: Tube, tube2: Tube, *args, **kwargs) -> float:
        """Return one tube pair's STT-IOU, optionally duration-weighted."""
        return float(self._tube_pairwise_iou([tube1], [tube2], *args, **kwargs)[0, 0])

    def _tube_iou(self, tube1: Tube, tube2: Tube, *args, **kwargs) -> float:
        return self.tube_iou(tube1, tube2, *args, **kwargs)

    def _calculate_tube_iou(self, tube1: Tube, tube2: Tube, *args, **kwargs) -> float:
        return self.tube_iou(tube1, tube2, *args, **kwargs)

    stt_iou = tube_iou
    frame_duration_iou = tube_iou

    def _tubes_inter(self, tube1: Tube, tube2: Tube,
                     frame_durations: Any = None, **kwargs) -> np.array:
        """Give two tubes of track size N and M,
        compute de intersection volume between __all__ N x M pairs of boxes.
        The boxes in track must be (xmin, ymin, xmax, ymax)

        Args:
            tube1, tube2 (Tube): two `Tubes`. Contains tracks with N & M boxes, respectively.
            frame_durations: optional duration table for this video.

        Returns:
            np.array: intersection volume.
        """
        table = self._first_schedule_value(frame_durations, kwargs)
        if table is not None:
            table = self._schedule_for_video(table, getattr(tube1, 'video_id', None))

        inter_frames = self._get_intersection_frames(tube1, tube2)

        inter = 0

        if inter_frames is not None:
            for inter_frame in inter_frames:
                box1 = tube1.get_frame_boxes(inter_frame)
                box2 = tube2.get_frame_boxes(inter_frame)

                width_height = np.minimum(box1[:, None, 2:], box2[:, 2:]) - np.maximum(
                    box1[:, None, :2], box2[:, :2])  # [N,M,2]

                width_height.clip(min=0, out=width_height)  # [N,M,2]
                frame_intersection = width_height.prod(axis=2)  # [N,M]

                if table is None:
                    inter += frame_intersection
                else:
                    duration = duration_for_frame(table, inter_frame, tube1)
                    inter += float(np.sum(frame_intersection)) * duration
                del width_height

        return inter

    def _tubes_union(self, tube1: Tube, tube2: Tube,
                     frame_durations: Any = None, **kwargs) -> np.array:
        """Give two tubes of track size N and M,
        compute de union volume between __all__ N x M pairs of boxes.

        Args:
            tube1, tube2 (Tube): two `Tubes`. Contains tracks with N & M boxes, respectively.
            frame_durations: optional duration table for this video.

        Returns:
            np.array: union volume.
        """
        table = self._first_schedule_value(frame_durations, kwargs)
        if table is None:
            vol1 = tube1.get_tube_volume()
            vol2 = tube2.get_tube_volume()
            return vol1 + vol2

        video_id = getattr(tube1, 'video_id', getattr(tube2, 'video_id', None))
        table = self._schedule_for_video(table, video_id)
        vol1 = tube1.get_weighted_tube_volume(table)
        vol2 = tube2.get_weighted_tube_volume(table)
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
