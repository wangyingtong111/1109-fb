from src.evaluators.coco_evaluator import (evaluate, evaluate_detections,
                                          get_coco_metrics,
                                          get_coco_summary)
from src.evaluators.pascal_voc_evaluator import get_pascalvoc_metrics

__all__ = [
    'get_coco_summary', 'get_coco_metrics', 'get_pascalvoc_metrics',
    'evaluate_boxes', 'evaluate_detections', 'evaluate'
]


def evaluate_boxes(groundtruth_bbs, detected_bbs, iou_threshold=0.5, **kwargs):
    """Convenience headless detection-evaluation entry point (COCO-style)."""
    return get_coco_metrics(groundtruth_bbs,
                            detected_bbs,
                            iou_threshold=iou_threshold,
                            **kwargs)
