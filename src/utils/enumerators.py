from enum import Enum


class MethodAveragePrecision(Enum):
    """
    Class representing if the coordinates are relative to the
    image size or are absolute values.

        Developed by: Rafael Padilla
        Last modification: Apr 28 2018
    """
    EVERY_POINT_INTERPOLATION = 1
    ELEVEN_POINT_INTERPOLATION = 2


class CoordinatesType(Enum):
    """
    Class representing if the coordinates are relative to the
    image size or are absolute values.

        Developed by: Rafael Padilla
        Last modification: Apr 28 2018
    """
    RELATIVE = 1
    ABSOLUTE = 2


class BBType(Enum):
    """
    Class representing if the bounding box is groundtruth or not.
    """
    GROUND_TRUTH = 1
    DETECTED = 2


class BBFormat(Enum):
    """
    Class representing the format of a bounding box.
    """
    XYWH = 1
    XYX2Y2 = 2
    PASCAL_XML = 3
    YOLO = 4


class FileFormat(Enum):
    ABSOLUTE_TEXT = 1
    PASCAL = 2
    LABEL_ME = 3
    COCO = 4
    CVAT = 5
    YOLO = 6
    OPENIMAGE = 7
    IMAGENET = 8
    UNKNOWN = 9


class DetectionOutcome(Enum):
    """
    Class representing the reason assigned to a detection when the per-detection
    details are requested in the PASCAL VOC evaluator.

        TP: the detection matched a not-yet-matched ground truth of the same class
            with IoU >= threshold.
        DUPLICATE: the detection overlaps (IoU >= threshold) a same-class ground
            truth that was already matched by another detection with higher
            confidence. It points to that other detection.
        LOCALIZATION: the highest IoU of the detection with a same-class ground
            truth is below the threshold (poor localization).
        WRONG_CLASS: the detection does not qualify as TP/duplicate/localization
            for its own class, but overlaps a ground truth of another class with
            IoU >= threshold.
        IGNORED: the detection is marked as ignored (or matches an ignored ground
            truth) and must not count as TP or FP.
    """
    TP = 'tp'
    DUPLICATE = 'duplicate'
    LOCALIZATION = 'localization'
    WRONG_CLASS = 'wrong_class'
    IGNORED = 'ignored'
