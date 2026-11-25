"""Continuous planar geometry helpers for rotated bounding boxes.

The functions in this module intentionally use open/continuous coordinates:
a ``w x h`` rectangle has area ``w * h`` and touching boundaries have zero
area.  This is the geometry used whenever a rotated box participates in an
evaluation, including rotated/axis-aligned mixed pairs.  It is separate from
the legacy inclusive pixel convention used by :class:`src.bounding_box.BoundingBox.iou`
for ordinary axis-aligned boxes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

EPS = 1e-12


def normalize_angle_degrees(angle, angle_unit=None, angle_rad=None):
    """Return an angle in degrees.

    ``angle`` is interpreted as degrees unless ``angle_unit`` explicitly says
    radians.  ``angle_rad`` is a convenience for callers that already have
    radians.
    """
    if angle_rad is not None:
        return math.degrees(float(angle_rad))
    if angle is None:
        return 0.0
    unit = 'degrees' if angle_unit is None else str(angle_unit).lower()
    value = float(angle)
    if unit in ('rad', 'radian', 'radians'):
        return math.degrees(value)
    if unit in ('deg', 'degree', 'degrees', ''):
        return value
    raise ValueError(f'Unknown angle unit: {angle_unit!r}')


def rotated_corners(cx, cy, w, h, angle_degrees=0.0):
    """Return the four corners of a rotated rectangle.

    The local side of length ``w`` points along angle ``angle_degrees``.
    Width/height swapping together with a 90 degree offset therefore produces
    exactly the same polygon.
    """
    theta = math.radians(float(angle_degrees))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    ux, uy = cos_t, sin_t
    # A 90 degree counter-clockwise rotation of u.
    vx, vy = -sin_t, cos_t
    hw = float(w) / 2.0
    hh = float(h) / 2.0
    return [
        (float(cx) - ux * hw - vx * hh, float(cy) - uy * hw - vy * hh),
        (float(cx) + ux * hw - vx * hh, float(cy) + uy * hw - vy * hh),
        (float(cx) + ux * hw + vx * hh, float(cy) + uy * hw + vy * hh),
        (float(cx) - ux * hw + vx * hh, float(cy) - uy * hw + vy * hh),
    ]


def polygon_area(polygon):
    """Absolute shoelace area of a polygon."""
    if len(polygon) < 3:
        return 0.0
    area2 = 0.0
    for i, point in enumerate(polygon):
        x1, y1 = point
        x2, y2 = polygon[(i + 1) % len(polygon)]
        area2 += float(x1) * float(y2) - float(y1) * float(x2)
    return abs(area2) * 0.5


def _signed_area2(polygon):
    area2 = 0.0
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        area2 += float(x1) * float(y2) - float(y1) * float(x2)
    return area2


def _cross(ax, ay, bx, by):
    return ax * by - ay * bx


def _line_intersection(s, e, a, b):
    """Intersection of segments/lines SE and AB used by polygon clipping."""
    sx, sy = s
    ex, ey = e
    ax, ay = a
    bx, by = b
    dx_se = ex - sx
    dy_se = ey - sy
    dx_ab = bx - ax
    dy_ab = by - ay
    denom = _cross(dx_ab, dy_ab, dx_se, dy_se)
    if abs(denom) < EPS:
        # Parallel boundary case.  Returning an endpoint is safe because the
        # contribution is zero-area/parallel and callers ignore tiny errors.
        return (ex, ey)
    t = -_cross(dx_ab, dy_ab, sx - ax, sy - ay) / denom
    # Numerical robustness only; a true crossing has 0 <= t <= 1.
    t = min(1.0, max(0.0, t))
    return (sx + t * dx_se, sy + t * dy_se)


def clip_polygon(subject, clip_poly):
    """Clip a convex subject polygon by a convex clip polygon."""
    subject = [(float(x), float(y)) for x, y in subject]
    clip_poly = [(float(x), float(y)) for x, y in clip_poly]
    if len(subject) < 3 or len(clip_poly) < 3:
        return []
    if abs(_signed_area2(clip_poly)) <= EPS:
        return []
    # Sutherland-Hodgman expects a consistent winding.  Use CCW.
    if _signed_area2(clip_poly) < 0:
        clip_poly = list(reversed(clip_poly))

    output = subject
    for edge_idx, a in enumerate(clip_poly):
        b = clip_poly[(edge_idx + 1) % len(clip_poly)]
        ex, ey = b[0] - a[0], b[1] - a[1]
        edge_len = math.hypot(ex, ey)
        if edge_len <= EPS:
            continue
        # Scale the boundary tolerance with edge length and typical float error.
        tolerance = 1e-10 * max(1.0, edge_len)
        if not output:
            break
        inputs = output
        output = []
        s = inputs[-1]
        for e in inputs:
            cross_e = _cross(ex, ey, e[0] - a[0], e[1] - a[1])
            cross_s = _cross(ex, ey, s[0] - a[0], s[1] - a[1])
            e_inside = cross_e >= -tolerance
            s_inside = cross_s >= -tolerance
            if e_inside:
                if not s_inside:
                    output.append(_line_intersection(s, e, a, b))
                output.append(e)
            elif s_inside:
                output.append(_line_intersection(s, e, a, b))
            s = e
    return output


def polygon_intersection_area(poly_a, poly_b):
    """Intersection area of two convex polygons."""
    area_a = polygon_area(poly_a)
    area_b = polygon_area(poly_b)
    if area_a <= EPS or area_b <= EPS:
        return 0.0
    intersection = clip_polygon(list(poly_a), list(poly_b))
    if len(intersection) < 3:
        return 0.0
    return max(0.0, polygon_area(intersection))


def _box_to_polygon(box):
    # Local import avoids a circular import at module import time.
    from src.bounding_box import BoundingBox

    if isinstance(box, BoundingBox):
        return box.get_continuous_polygon(), float(box.get_continuous_area())

    if isinstance(box, dict):
        if 'polygon' in box:
            poly = [tuple(p) for p in box['polygon']]
            return poly, polygon_area(poly)
        cx = box.get('cx', box.get('center_x', box.get('x', 0.0)))
        cy = box.get('cy', box.get('center_y', box.get('y', 0.0)))
        w = box.get('w', box.get('width', 0.0))
        h = box.get('h', box.get('height', 0.0))
        angle = normalize_angle_degrees(box.get('angle', 0.0), box.get('angle_unit'),
                                       box.get('angle_rad'))
        return rotated_corners(cx, cy, w, h, angle), abs(float(w) * float(h))

    if isinstance(box, Sequence) and not isinstance(box, (str, bytes)):
        values = [float(v) for v in box]
        if len(values) == 5:
            cx, cy, w, h, angle = values
            return rotated_corners(cx, cy, w, h, angle), abs(w * h)
        if len(values) == 4:
            # Raw four-tuples are interpreted as x1, y1, x2, y2 to match the
            # common BoundingBox XYX2Y2 export.
            x1, y1, x2, y2 = values
            poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            return poly, abs((x2 - x1) * (y2 - y1))

    raise TypeError(f'Unsupported geometry box: {box!r}')


def geometry_iou(box_a, box_b):
    """Continuous polygon IoU for rotated or axis-aligned boxes."""
    poly_a, area_a = _box_to_polygon(box_a)
    poly_b, area_b = _box_to_polygon(box_b)
    if area_a <= EPS or area_b <= EPS:
        # A measure-zero rectangle has no positive continuous overlap.  This
        # explicit boundary avoids 0/0 and prevents degenerate boxes from
        # becoming matches in mixed evaluations.
        return 0.0
    inter = polygon_intersection_area(poly_a, poly_b)
    union = area_a + area_b - inter
    if union <= EPS:
        return 0.0
    iou = inter / union
    # Remove tiny floating-point excursions outside [0, 1].
    return min(1.0, max(0.0, float(iou)))


rotated_iou = geometry_iou
polygon_iou = geometry_iou
