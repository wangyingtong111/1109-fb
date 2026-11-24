"""Shared geometric convention for comparing boxes of different origins.

Bounding boxes found in a single detection report may come from two different
geometric conventions:

* continuous half-open intervals, e.g. COCO boxes whose coordinates can be
  floating point: ``[x1, x2) x [y1, y2)``;
* inclusive integer pixel spans, e.g. PASCAL VOC boxes where
  ``[3, 0, 3, 0]`` represents the whole pixel at column 3, row 0.

The functions in this module define the *shared* geometry convention used by
the evaluation entry points that explicitly opt in to it. Every box is
interpreted as a continuous half-open region based on its own per-box
geometry marker (see :class:`src.utils.enumerators.BBGeometry`):

* ``BBGeometry.CONTINUOUS`` boxes are used as-is: ``[x1, y1, x2, y2]``;
* ``BBGeometry.PIXEL_CLOSED`` boxes are non-destructively mapped to the
  continuous half-open region ``[x1, x2 + 1) x [y1, y2 + 1)``.

Under this unified convention, regions that only touch at an edge share no
area, hence their intersection (and IoU) is zero.

The conversion is applied on the fly to copies of the coordinates: the input
:class:`src.bounding_box.BoundingBox` objects (and their stored coordinates)
are never mutated, so calling the evaluators repeatedly is idempotent and the
inclusive pixel edge is never counted twice.
"""

from src.bounding_box import BBFormat
from src.utils.enumerators import BBGeometry


def get_continuous_coordinates(box):
    """Return the box coordinates as a continuous, half-open XYXY region.

    Parameters
    ----------
    box : src.bounding_box.BoundingBox
        Bounding box whose per-box geometry convention is read through
        :meth:`BoundingBox.get_geometry`.

    Returns
    -------
    tuple
        Four floats ``(x1, y1, x2, y2)`` describing the half-open region
        ``[x1, x2) x [y1, y2)``. For pixel-closed boxes the inclusive outer
        edge is expanded by one unit, so a one-pixel box ``[3, 0, 3, 0]``
        becomes ``(3.0, 0.0, 4.0, 1.0)``.

    Notes
    -----
    The input box is never modified; a fresh tuple is returned.
    """
    x1, y1, x2, y2 = box.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
    geometry = box.get_geometry()
    if geometry == BBGeometry.PIXEL_CLOSED:
        # inclusive pixel span [x1, x2] maps to the half-open [x1, x2 + 1)
        x2 = x2 + 1
        y2 = y2 + 1
    return (float(x1), float(y1), float(x2), float(y2))


def get_continuous_area(box):
    """Area of the box under the shared (continuous, half-open) convention."""
    x1, y1, x2, y2 = get_continuous_coordinates(box)
    return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


def shared_iou(box_a, box_b):
    """IoU of two boxes under the shared geometry convention.

    Each box is interpreted through its own per-box geometry marker and
    converted on the fly to a continuous half-open region. Boxes that only
    touch at an edge have zero intersection, and zero-area boxes (e.g. a
    continuous zero-width box) have zero IoU.

    Parameters
    ----------
    box_a, box_b : src.bounding_box.BoundingBox
        Bounding boxes potentially built with different geometry conventions.

    Returns
    -------
    float
        Intersection over union in the interval ``[0, 1]``.
    """
    xa, ya, x2a, y2a = get_continuous_coordinates(box_a)
    xb, yb, x2b, y2b = get_continuous_coordinates(box_b)

    # innermost edges of the overlap
    xi = max(xa, xb)
    yi = max(ya, yb)
    x2i = min(x2a, x2b)
    y2i = min(y2a, y2b)

    # half-open intervals: a mere edge contact has zero width/height
    inter_w = max(x2i - xi, 0.0)
    inter_h = max(y2i - yi, 0.0)
    inter_area = inter_w * inter_h

    area_a = max(x2a - xa, 0.0) * max(y2a - ya, 0.0)
    area_b = max(x2b - xb, 0.0) * max(y2b - yb, 0.0)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        # degenerate boxes (zero area) cannot overlap over any region
        return 0.0
    return inter_area / union_area
