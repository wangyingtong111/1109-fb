from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np

from src.utils.general_utils import convert_box_xywh2xyxy


class FrameScheduleError(KeyError, ValueError):
    """Raised when an explicit frame timetable does not cover a used frame."""

    def __init__(self, video_id: Any = None, frame: Any = None, missing_frames: Optional[list] = None):
        self.video_id = video_id
        self.frame = frame
        if missing_frames is None and frame is not None:
            missing_frames = [frame]
        self.missing_frames = list(missing_frames or [])

        if video_id is None:
            message = f"Frame timetable is missing used frame(s): {self.missing_frames}"
        else:
            message = (
                f"Frame timetable for video {video_id!r} is missing used frame(s): "
                f"{self.missing_frames}"
            )
        super().__init__(message)


MissingFrameError = FrameScheduleError
MissingFrameDurationError = FrameScheduleError
FrameTimetableError = FrameScheduleError


def normalize_frame_key(frame: Any) -> Any:
    """Normalize JSON/NumPy frame keys so 0, '0' and np.int64(0) compare equal."""
    if isinstance(frame, np.generic):
        frame = frame.item()
    if isinstance(frame, str):
        try:
            return int(frame)
        except ValueError:
            return frame
    if isinstance(frame, (int, np.integer)):
        return int(frame)
    return frame


def _duration_value(value: Any) -> float:
    if callable(value):
        value = value()
    if isinstance(value, Mapping):
        for key in ("duration", "duration_seconds", "seconds", "sec", "length", "weight", "dt"):
            if key in value:
                return float(value[key])
        if "start" in value and "end" in value:
            return float(value["end"]) - float(value["start"])
        if "start_time" in value and "end_time" in value:
            return float(value["end_time"]) - float(value["start_time"])
        raise ValueError(f"Unrecognized frame duration specification: {value!r}")
    if isinstance(value, (tuple, list, np.ndarray)) and len(value) == 2:
        # Explicit [start, end] interval.  Frame/duration pairs are handled
        # before this helper when normalizing a sequence of records.
        return float(value[1]) - float(value[0])
    return float(value)


def _is_scalar_duration(value: Any) -> bool:
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        return True
    if isinstance(value, Mapping):
        duration_keys = (
            "duration", "duration_seconds", "seconds", "sec", "length",
            "weight", "dt", "start", "start_time",
        )
        return any(key in value for key in duration_keys) and not (
            "frames" in value or "frame_ids" in value
        )
    return False


def normalize_frame_table(schedule: Any) -> Any:
    """Normalize one video's frame -> duration table.

    A callable is returned unchanged.  Plain numeric sequences are indexed by
    frame number, while record and ``(frame, duration)`` sequences are converted
    to dictionaries.
    """
    if schedule is None or callable(schedule):
        return schedule
    if isinstance(schedule, np.ndarray):
        schedule = schedule.tolist()

    if isinstance(schedule, Mapping):
        if "frames" in schedule and any(k in schedule for k in
                                        ("durations", "duration", "frame_durations", "seconds")):
            frames = schedule["frames"]
            durations = schedule.get(
                "durations",
                schedule.get("frame_durations", schedule.get("duration", schedule.get("seconds"))),
            )
            return {
                normalize_frame_key(frame): _duration_value(duration)
                for frame, duration in zip(frames, durations)
            }
        if "frame_ids" in schedule and any(k in schedule for k in ("durations", "frame_durations")):
            return {
                normalize_frame_key(frame): _duration_value(duration)
                for frame, duration in zip(schedule["frame_ids"], schedule["durations"])
            }

        result = {}
        for frame, value in schedule.items():
            if frame in ("video_id", "id", "name"):
                continue
            result[normalize_frame_key(frame)] = _duration_value(value)
        return result

    if isinstance(schedule, Sequence) and not isinstance(schedule, (str, bytes)):
        if len(schedule) == 0:
            return {}

        if all(isinstance(item, Mapping) for item in schedule):
            result = {}
            for item in schedule:
                frame = None
                for key in ("frame", "frame_id", "frame_idx", "index", "id"):
                    if key in item:
                        frame = item[key]
                        break
                if frame is None:
                    raise ValueError(f"Frame duration record lacks a frame key: {item!r}")
                result[normalize_frame_key(frame)] = _duration_value(item)
            return result

        if all(isinstance(item, (tuple, list)) and len(item) == 2 for item in schedule):
            # Distinguish [(frame, duration), ...] from [[start, end], ...] by
            # treating the first column as frame identifiers.
            return {
                normalize_frame_key(frame): _duration_value(duration)
                for frame, duration in schedule
            }

        return {index: _duration_value(value) for index, value in enumerate(schedule)}

    return schedule


def duration_for_frame(table: Any, frame: Any, tube: Optional["Tube"] = None) -> float:
    """Return the positive/non-negative duration assigned to ``frame``."""
    if table is None:
        return 1.0

    frame_key = normalize_frame_key(frame)

    if callable(table):
        try:
            duration = table(frame_key)
        except TypeError:
            duration = table(frame)
        duration = _duration_value(duration)
    else:
        normalized = normalize_frame_table(table)
        duration = None
        if isinstance(normalized, Mapping):
            for key in (frame_key, normalize_frame_key(frame)):
                if key in normalized:
                    duration = normalized[key]
                    break
            if duration is None and isinstance(frame_key, int):
                for key, value in normalized.items():
                    if normalize_frame_key(key) == frame_key:
                        duration = value
                        break
        elif isinstance(normalized, Sequence) and not isinstance(normalized, (str, bytes)):
            if isinstance(frame_key, int) and 0 <= frame_key < len(normalized):
                duration = normalized[frame_key]
            elif tube is not None:
                positions = [
                    index for index, tube_frame in enumerate(tube.get_frames())
                    if normalize_frame_key(tube_frame) == frame_key
                ]
                if positions and positions[0] < len(normalized):
                    duration = normalized[positions[0]]

        if (duration is None and tube is not None
                and not isinstance(table, Mapping)
                and not callable(table)):
            # A bare numeric list may be aligned with the tube's track order
            # rather than absolute frame numbers.  Explicit (frame, duration)
            # pair lists must still report a genuinely missing frame.
            raw_table = table
            if isinstance(raw_table, np.ndarray):
                raw_table = raw_table.tolist()
            if (isinstance(raw_table, Sequence) and not isinstance(raw_table, (str, bytes))
                    and len(raw_table) > 0
                    and all(_is_scalar_duration(item) for item in raw_table)):
                try:
                    frames = [normalize_frame_key(f) for f in tube.get_frames()]
                    position = frames.index(frame_key)
                    if position < len(raw_table):
                        duration = raw_table[position]
                except ValueError:
                    pass

        if duration is None:
            video_id = getattr(tube, "video_id", None)
            raise FrameScheduleError(video_id=video_id, frame=frame)
        duration = _duration_value(duration)

    duration = float(duration)
    if not np.isfinite(duration) or duration < 0:
        raise ValueError(f"Frame duration must be a finite non-negative number, got {duration!r}")
    return duration


class Tube(object):
    """Tube object
    """
    def __init__(self,
                 category_id: int,
                 video_id: int,
                 track: list,
                 frame_durations: Any = None,
                 **kwargs: Any) -> None:
        """class constructor

        Args:
            category_id (int): class category of corresponding tube
            video_id (int): video id of corresponding tube
            track (list of dicts): list of dictionaries with keys "frame" and "bbox"
            frame_durations (optional): frame duration table used by duration-aware
                IoU.  Evaluator-owned per-video tables take precedence over this value.
        """
        self.category_id = category_id
        self.video_id = video_id
        self.confidence = None
        self.frame_durations = frame_durations

        self.track = {attr: np.array([det[attr] for det in track]) for attr in track[0]}

        if 'confidence' in self.track.keys():
            self.confidence = self.__compute_tube_confidence(self.track['confidence'])

        self.volume = self.__compute_tube_volume()

        self.__kwargs = kwargs
        for k, v in kwargs.items():
            super().__setattr__(k, v)

        # convert tube boxes from xywh to xyxy format
        boxes = convert_box_xywh2xyxy(self.get_boxes())
        self.track['bbox'] = boxes

        duration_table = self.get_duration_table()
        if duration_table is not None:
            # Explicit duration metadata turns the stored volume into a
            # time-weighted volume.  The default path keeps the original value.
            self.volume = self.get_weighted_tube_volume(duration_table)

    def __len__(self) -> int:
        return len(self.track)

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "video_id={}, ".format(self.video_id)
        s += "category_id={}, ".format(self.category_id)
        s += "tube_confidence={:.2f}, ".format(self.confidence)
        s += "{}, ".format("".join(f"{k}={v}" for k, v in self.__kwargs.items()))
        s += "track=[{}]".format(", ".join(f"{k}: {v}" for k, v in self.track.items()))
        s += ")"

        return s

    def __compute_tube_confidence(self, frames_confidences: list) -> float:
        """Compute tube confidence. It is the mean of individual bboxes along the frames.

        Args:
            frames_confidences (list): list containing the frames confidences

        Returns:
            numpy: the tube confidence
        """
        return np.mean(frames_confidences)

    def get_frames(self) -> list:
        """Return a list of frames of a given track.

        Returns:
            list: list of frames of tube.
        """
        return self.track['frame']

    def get_boxes(self) -> list:
        """Return a list of bboxes of a given track.

        Returns:
            list: list of bboxes of tube.
        """
        return self.track['bbox']

    def get_frame_boxes(self, frame_idx):
        """return the boxes of frame `frame_idx`

        Args:
            frame_idx (int): frame index

        Returns:
            list: list of bounding boxes of the `frame_idx` frame
        """
        frames = self.get_frames()
        indexes = [i for i, e in enumerate(frames) if e == frame_idx]

        boxes = self.get_boxes()[indexes]

        return boxes

    def __compute_tube_volume(self) -> float:
        """Compute tube volume. It is the summation of pixels of all bouding boxes that the tube contains.

        Returns:
            float: tube volume
        """
        boxes = self.get_boxes()
        areas = np.prod(boxes[:, 2:], axis=1)
        vol = areas.sum()
        return vol

    def get_tube_volume(self, frame_durations: Any = None, **kwargs: Any) -> float:
        """return the tube volume previously computed in self.__compute_tube_volume

        Args:
            frame_durations: optional duration table. When supplied, return the
                area-by-duration volume instead of the stored equal-frame volume.

        Returns:
            float: tube volume
        """
        if frame_durations is None:
            for name in ("timetable", "frame_timetable", "durations", "duration_table"):
                if kwargs.get(name) is not None:
                    frame_durations = kwargs[name]
                    break
        if frame_durations is not None:
            return self.get_weighted_tube_volume(frame_durations)
        return self.volume

    def get_frame_areas(self) -> np.ndarray:
        """Return geometric area of every converted XYX2Y2 box."""
        boxes = self.get_boxes()
        if boxes.size == 0:
            return np.zeros((0,), dtype=float)
        widths_heights = boxes[:, 2:] - boxes[:, :2]
        widths_heights = np.maximum(widths_heights, 0)
        return np.prod(widths_heights, axis=1)

    def get_duration_table(self, frame_durations: Any = None) -> Any:
        """Return the explicitly supplied duration table for this tube, if any."""
        if frame_durations is not None:
            table = frame_durations
        else:
            table = None
            for name in ("frame_durations", "frame_duration", "duration_map",
                         "frame_duration_map", "duration_table", "frame_table",
                         "frame_timetable", "timetable", "time_table",
                         "durations", "duration", "frame_lengths",
                         "frame_weights"):
                if hasattr(self, name):
                    value = getattr(self, name)
                    if value is not None:
                        # A bare scalar "duration" may be a total clip length,
                        # not a frame table; only accept table-like values.
                        if name == "duration" and _is_scalar_duration(value):
                            continue
                        table = value
                        break
            if table is None and isinstance(getattr(self, "track", None), Mapping):
                for name in ("duration", "durations", "frame_duration", "seconds"):
                    if name in self.track:
                        table = self.track[name]
                        break

        # Accept a per-video mapping when the caller gives a Tube its own batch
        # timetable directly.
        if isinstance(table, Mapping) and self.video_id is not None:
            video_key = normalize_frame_key(self.video_id)
            for key in (self.video_id, video_key, str(self.video_id)):
                if key in table and not _is_scalar_duration(table[key]):
                    return table[key]
        return table

    def get_weighted_tube_volume(self, frame_durations: Any = None) -> float:
        """Compute area * frame-duration volume.

        Without a duration table this is equivalent to the original frame-count
        volume (each frame has weight one).
        """
        table = self.get_duration_table(frame_durations)
        if table is None:
            return float(self.get_tube_volume())

        frames = self.get_frames()
        areas = self.get_frame_areas()
        volume = 0.0
        for frame, area in zip(frames, areas):
            volume += float(area) * duration_for_frame(table, frame, self)
        return volume

    def get_frame_duration_map(self, frame_durations: Any = None) -> dict:
        """Return this tube's frame IDs mapped to their durations."""
        table = self.get_duration_table(frame_durations)
        if table is None:
            return {}
        return {
            normalize_frame_key(frame): duration_for_frame(table, frame, self)
            for frame in self.get_frames()
        }

    def iou(self, other: "Tube", frame_durations: Any = None,
            timetable: Any = None, frame_timetable: Any = None,
            **kwargs: Any) -> float:
        """Pairwise tube IoU, optionally weighted by this video's frame timetable."""
        # Import here to preserve evaluator -> Tube import direction.
        from src.evaluators.tube_evaluator import TubeEvaluator

        table = None
        for value in (frame_durations, timetable, frame_timetable,
                      kwargs.get("durations"), kwargs.get("frame_duration")):
            if value is not None:
                table = value
                break

        evaluator = TubeEvaluator.__new__(TubeEvaluator)
        result = evaluator._tube_pairwise_iou([self], [other], frame_durations=table)
        return float(result[0, 0])

    @classmethod
    def tube_iou(cls, tube1: "Tube", tube2: "Tube", frame_durations: Any = None,
                 timetable: Any = None, **kwargs: Any) -> float:
        """Functional alias for :meth:`iou`."""
        return tube1.iou(
            tube2, frame_durations=frame_durations, timetable=timetable, **kwargs
        )

    __repr__ = __str__


def tube_iou(tube1: Tube, tube2: Tube, frame_durations: Any = None,
             timetable: Any = None, **kwargs: Any) -> float:
    """Duration-aware tube IoU functional API."""
    return Tube.tube_iou(
        tube1, tube2, frame_durations=frame_durations,
        timetable=timetable, **kwargs
    )


def frame_weighted_tube_iou(tube1: Tube, tube2: Tube,
                            frame_timetable: Any = None) -> float:
    """Alias emphasizing a per-video frame timetable."""
    return tube_iou(tube1, tube2, frame_durations=frame_timetable)
