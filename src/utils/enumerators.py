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
    # Center-x, center-y, width, height, angle (degrees by default).
    CXCYWHA = 5
    CXCYWH = 6
    CXCYWHA_RAD = 7

    # Common aliases used by rotated-object-detection code.
    CXCYWH_ANGLE = 5
    CX_CY_W_H_A = 5
    CX_CY_W_H_ANGLE = 5
    XYWHA = 5
    X_Y_W_H_A = 5
    ROTATED_XYWH = 5
    ROTATED_CXCYWH = 5
    ROTATED_CENTER = 5
    OBB = 5
    ROTATED = 5
    RBOX = 5
    CXCYWHAR = 7
    XYWHA_RAD = 7
    RBOX_RAD = 7
    CENTER_XYWH = 6
    CX_CY_W_H = 6

    @classmethod
    def parse(cls, value):
        """Return a BBFormat from an enum, integer, or common string name."""
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            return cls(value)
        if isinstance(value, str):
            key = ''.join(ch for ch in value.strip().upper() if ch.isalnum())
            aliases = {
                'CXCYWHA': cls.CXCYWHA,
                'CXCYWHANGLE': cls.CXCYWHA,
                'CXCYWHARAD': cls.CXCYWHA_RAD,
                'CXCYWHAR': cls.CXCYWHA_RAD,
                'XYWHARAD': cls.CXCYWHA_RAD,
                'RBOXRAD': cls.CXCYWHA_RAD,
                'XYWHA': cls.CXCYWHA,
                'RBOX': cls.CXCYWHA,
                'ROTATED': cls.CXCYWHA,
                'ROTATEDXYWH': cls.CXCYWHA,
                'ROTATEDCXCYWH': cls.CXCYWHA,
                'ROTATEDCENTER': cls.CXCYWHA,
                'OBB': cls.CXCYWHA,
                'XYWHAANGLE': cls.CXCYWHA,
                'CXCYWH': cls.CXCYWH,
                'CENTERXYWH': cls.CXCYWH,
                'XYWH': cls.XYWH,
                'XYX2Y2': cls.XYX2Y2,
                'PASCALXML': cls.PASCAL_XML,
                'YOLO': cls.YOLO,
            }
            if key in aliases:
                return aliases[key]
        raise ValueError(f'Unknown bounding box format: {value!r}')


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
