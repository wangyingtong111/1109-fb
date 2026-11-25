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


class CoordinatesPrecision(Enum):
    """
    Class representing how relative (normalized) coordinates are converted into
    absolute ones.

    This only affects boxes informed in relative coordinates (e.g. the
    (x_center, y_center, width, height) output of a detection model). Boxes
    already informed in absolute coordinates are always stored as they are.

    EXACT
        The normalized values are converted without rounding, so sub-pixel
        geometry is kept all the way to the matching/scoring step. On a
        100x50 image, a box with center x=.5 and relative width .005 has the
        x borders 49.75 and 50.25, and its relative width round-trips back to
        .005.
    QUANTIZED
        Legacy behaviour: the absolute corners are rounded to the nearest
        integer pixel, as the original BoundingBox pixel conversion did.
        Small boxes may then collapse onto integer corners.
    """
    EXACT = 1
    QUANTIZED = 2


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
