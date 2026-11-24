import json

from src.bounding_box import BBFormat, BBType, BoundingBox
from src.evaluators.coco_evaluator import (
    _compute_ious,
    _evaluate_image,
    _group_detections,
    _jaccard,
    get_coco_detection_statuses,
    get_coco_metrics,
    get_coco_summary,
)
from src.utils.converter import coco2bb


def make_box(rect, score=None, label="x", image="im", **metadata):
    kwargs = {}
    if score is not None:
        kwargs["bb_type"] = BBType.DETECTED
        kwargs["confidence"] = score
    box = BoundingBox(image, label, rect, format=BBFormat.XYWH, **kwargs)
    # The evaluation contract must also recognize markers attached by callers.
    for key, value in metadata.items():
        setattr(box, key, value)
    return box


def crowd_scene():
    regular = (0, 0, 4, 4)
    crowd = (-1, -1, 30, 30)
    groundtruth = [make_box(regular), make_box(crowd, iscrowd=True)]
    detections = [
        make_box(regular, 0.9),
        make_box((10, 10, 2, 2), 0.8),
        make_box((14, 14, 4, 4), 0.7),
    ]
    return groundtruth, detections


def test_crowd_absorbs_multiple_predictions_without_fp_or_positive():
    groundtruth, detections = crowd_scene()

    result = get_coco_metrics(groundtruth, detections, iou_threshold=0.5)["x"]

    assert result["total positives"] == 1
    assert result["TP"] == 1
    assert result["FP"] == 0
    assert result["ignored"] == 2
    assert result["statuses"] == ["TP", "ignored", "ignored"]
    assert result["detection_statuses"] == ["TP", "ignored", "ignored"]
    assert [record["status"] for record in result["records"]] == [
        "TP",
        "ignored",
        "ignored",
    ]
    assert all(record["reason"] == "crowd" for record in result["records"][1:])


def test_pairwise_overlap_uses_detection_coverage_for_crowd():
    groundtruth, detections = crowd_scene()
    overlaps = _compute_ious(detections, groundtruth)

    assert overlaps[0, 0] == 1.0
    assert overlaps[1, 1] == 1.0
    assert overlaps[2, 1] == 1.0
    # Standard IoU with the large crowd would be close to zero; the crowd
    # coverage convention must therefore be exposed by _compute_ious too.
    assert overlaps[1, 1] > _jaccard(detections[1], groundtruth[1])


def test_crowd_matching_allows_many_dets_but_regular_gt_only_once():
    groundtruth, detections = crowd_scene()
    grouped = _group_detections(detections, groundtruth)
    key = ("im", "x")
    ious = _compute_ious(**grouped[key])

    report = _evaluate_image(
        grouped[key]["dt"], grouped[key]["gt"], ious, 0.5, 100, (0, float("inf"))
    )

    assert report["NP"] == 1
    assert report["dt_ignore"] == [False, True, True]
    assert report["statuses"] == ["TP", "ignored", "ignored"]
    # Both crowd predictions are associated with the same crowd GT.
    assert sorted(report["crowd_matches"].values()) == [[1, 2]]
    # The regular GT has only one winning prediction.
    assert report["dtm"][0] == 0
    assert report["gtm"][0] == 0

    ordered = get_coco_detection_statuses(groundtruth, detections)
    assert [record["status"] for record in ordered] == [
        "TP",
        "ignored",
        "ignored",
    ]
    assert [record["input_index"] for record in ordered] == [0, 1, 2]


def test_ordinary_no_crowd_scores_unchanged():
    gt = [BoundingBox("im", "x", (0, 0, 16, 16))]
    det = [
        BoundingBox(
            "im",
            "x",
            (0, 0, 16, 16),
            bb_type=BBType.DETECTED,
            confidence=0.9,
        )
    ]

    metrics = get_coco_metrics(gt, det)["x"]
    assert metrics["total positives"] == 1
    assert metrics["TP"] == 1
    assert metrics["FP"] == 0
    assert metrics["statuses"] == ["TP"]
    assert get_coco_summary(gt, det)["AP50"] == 1


def test_coco_importer_reads_iscrowd(tmp_path):
    annotation_path = tmp_path / "crowd.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "im.jpg", "width": 100, "height": 100}
                ],
                "categories": [{"id": 7, "name": "x"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 7,
                        "bbox": [0, 0, 4, 4],
                        "iscrowd": 0,
                    },
                    {
                        "id": 2,
                        "image_id": 1,
                        "category_id": 7,
                        "bbox": [-1, -1, 30, 30],
                        "iscrowd": 1,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    gts = coco2bb(str(annotation_path))
    assert [box.iscrowd for box in gts] == [False, True]
    assert [box.is_crowd for box in gts] == [False, True]
