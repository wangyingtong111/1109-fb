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


class BBGeometry(Enum):
    """
    Class representing the geometric convention of a bounding box.

    This convention describes how the four stored XYXY coordinates should be
    interpreted geometrically. It is only used by evaluation entry points that
    explicitly opt in to the shared geometry convention; the legacy evaluators
    keep their historical behaviour untouched.

    CONTINUOUS
        Continuous, half-open intervals [x1, x2) x [y1, y2). Coordinates may
        be floating point. Two boxes that only touch at an edge do not share
        any area, so their intersection (and IoU) is 0.
    PIXEL_CLOSED
        Inclusive pixel spans, e.g. [3, 0, 3, 0] denotes the single pixel at
        column 3, row 0. The coordinates are integer and both extremes are
        included. Such boxes are interpreted as the continuous half-open box
        [x1, x2 + 1) x [y1, y2 + 1) inside the shared geometry convention.
    """
    CONTINUOUS = 1
    PIXEL_CLOSED = 2


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
