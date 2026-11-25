import pytest

from src.evaluators.tube_evaluator import TubeEvaluator
from src.tube import FrameScheduleError, Tube


def make_tube(video_id, frames):
    return Tube(
        category_id=1,
        video_id=video_id,
        track=[{"frame": frame, "bbox": [0, 0, 2, 2]} for frame in frames],
    )


@pytest.fixture()
def evaluator():
    return TubeEvaluator.__new__(TubeEvaluator)


def test_missing_timetable_keeps_equal_frame_iou(evaluator):
    values = [
        float(evaluator._tube_pairwise_iou(
            [make_tube(video_id, [0])],
            [make_tube(video_id, [0, 1])],
        )[0, 0])
        for video_id in (1, 2)
    ]
    assert values == pytest.approx([0.5, 0.5])


def test_per_video_frame_durations(evaluator):
    timetables = {
        1: {0: 1, 1: 3},
        2: {0: 3, 1: 1},
    }

    values = [
        float(evaluator._tube_pairwise_iou(
            [make_tube(video_id, [0])],
            [make_tube(video_id, [0, 1])],
            frame_durations=timetables,
        )[0, 0])
        for video_id in (1, 2)
    ]

    assert values == pytest.approx([0.25, 0.75])


def test_explicit_timetable_missing_used_frame_is_reported(evaluator):
    timetables = {2: {0: 3, 1: 1}}
    preds = [make_tube(2, [0])]
    gts = [make_tube(2, [0, 3])]

    assert evaluator.missing_frames(
        preds, gts, frame_timetables=timetables, video_id=2
    ) == [3]

    with pytest.raises((KeyError, ValueError), match="3"):
        evaluator._tube_pairwise_iou(
            preds, gts, frame_durations=timetables
        )

    with pytest.raises(FrameScheduleError):
        evaluator._tube_pairwise_iou(preds, gts, frame_durations=timetables)


def test_absent_box_is_not_interpolated(evaluator):
    # Frame 1 exists in the reference but the prediction has no box there.
    # Its full duration therefore remains unmatched; no interpolated box is made.
    iou = evaluator.tube_iou(
        make_tube(1, [0]),
        make_tube(1, [0, 1]),
        frame_durations={1: {0: 1, 1: 3}},
    )
    assert iou == pytest.approx(0.25)
