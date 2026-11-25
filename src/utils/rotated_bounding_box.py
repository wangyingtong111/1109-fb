"""Public alias for rotated-box geometry and constructor."""

from src.bounding_box import (BoundingBox, OBBBoundingBox, RotatedBBox,
                              RotatedBoundingBox, RotatedBox,
                              RotatingBoundingBox, rotated_iou)
from src.utils.rotated_geometry import (geometry_iou,
                                        polygon_intersection_area,
                                        rotated_corners)

__all__ = [
    'BoundingBox', 'RotatedBoundingBox', 'RotatingBoundingBox', 'RotatedBBox',
    'RotatedBox', 'OBBBoundingBox', 'rotated_iou', 'geometry_iou',
    'rotated_corners', 'polygon_intersection_area'
]
