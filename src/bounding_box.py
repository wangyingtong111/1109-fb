import math
from math import isclose

from src.utils.general_utils import (convert_to_absolute_values,
                                     convert_to_relative_values)
from src.utils.rotated_geometry import (geometry_iou,
                                        normalize_angle_degrees,
                                        polygon_intersection_area,
                                        rotated_corners)

from .utils.enumerators import BBFormat, BBType, CoordinatesType

_ROTATED_FORMATS = {
    BBFormat.CXCYWHA,
    BBFormat.CXCYWH,
    BBFormat.CXCYWHA_RAD,
}
_RADIAN_FORMATS = {BBFormat.CXCYWHA_RAD}


def _normalize_coordinates(coordinates):
    """Flatten common rotated coordinate containers to ``(cx,cy,w,h[,angle])``."""
    if coordinates is None:
        return ()
    # numpy arrays and other iterables become tuples without importing numpy.
    coordinates = tuple(coordinates)

    def _pair(value):
        if value is None or isinstance(value, (str, bytes)):
            return None
        try:
            values = tuple(value)
        except TypeError:
            return None
        if len(values) == 2:
            return float(values[0]), float(values[1])
        return None

    if len(coordinates) == 3:
        center = _pair(coordinates[0])
        size = _pair(coordinates[1])
        if center is not None and size is not None:
            return (center[0], center[1], size[0], size[1], float(coordinates[2]))
    if len(coordinates) == 4:
        size = _pair(coordinates[2])
        center = _pair(coordinates[0]) if not isinstance(coordinates[0], (int, float)) else None
        if size is not None and center is not None:
            return (center[0], center[1], size[0], size[1], float(coordinates[3]))
        if size is not None and isinstance(coordinates[0], (int, float)):
            return (float(coordinates[0]), float(coordinates[1]), size[0], size[1],
                    float(coordinates[3]))
    return coordinates


class BoundingBox:
    """ Class representing a bounding box. """
    def __init__(self,
                 image_name,
                 class_id=None,
                 coordinates=None,
                 type_coordinates=CoordinatesType.ABSOLUTE,
                 img_size=None,
                 bb_type=BBType.GROUND_TRUTH,
                 confidence=None,
                 format=BBFormat.XYWH,
                 angle=None,
                 angle_unit=None,
                 angle_rad=None,
                 rotation=None,
                 rotated=None,
                 center=None,
                 width=None,
                 height=None,
                 **kwargs):
        """ Constructor.

        Rotated boxes can be supplied as ``(center_x, center_y, width, height,
        angle)`` with ``format=BBFormat.CXCYWHA``.  ``angle`` is in degrees by
        default; pass ``angle_unit='radians'`` or ``angle_rad=...`` for radians.
        A four-coordinate center box can alternatively be supplied with
        ``angle=...`` or ``format=BBFormat.CXCYWH``.
        """

        # Compatibility aliases for callers/tests using slightly different
        # keyword names.
        if angle is None:
            angle = rotation
        if angle is None:
            angle = kwargs.pop('angle_degrees', kwargs.pop('angle_deg', None))
        if angle is None:
            angle = kwargs.pop('rotation_angle', kwargs.pop('theta', None))
        if angle_unit is None:
            angle_unit = kwargs.pop('angle_units', kwargs.pop('unit', None))
        if angle_unit is None:
            angle_unit = kwargs.pop('angle_format', None)
        if angle_rad is None:
            angle_rad = kwargs.pop('angle_radians',
                                   kwargs.pop('theta_rad', kwargs.pop('rotation_rad', None)))
        if rotated is None:
            rotated = kwargs.pop('is_rotated', None)
        for format_alias in ('bb_format', 'box_format', 'coordinate_format', 'coord_format'):
            if format_alias in kwargs:
                format = kwargs.pop(format_alias)

        if not isinstance(format, BBFormat):
            format = BBFormat.parse(format)
        if angle_unit is None and format in _RADIAN_FORMATS:
            angle_unit = 'radians'

        if coordinates is None and center is not None:
            coordinates = (center[0], center[1], width, height)
        if coordinates is None and width is not None and height is not None:
            # Allow BoundingBox(img, cls, cx=..., cy=..., width=..., ...).
            coordinates = (kwargs.pop('cx', kwargs.pop('center_x', 0.0)),
                           kwargs.pop('cy', kwargs.pop('center_y', 0.0)),
                           width, height)

        coordinates = _normalize_coordinates(coordinates)
        coordinate_angle = coordinates[4] if len(coordinates) >= 5 else None
        if angle is None:
            angle = coordinate_angle

        self._angle = normalize_angle_degrees(angle, angle_unit, angle_rad)
        self._rotation_explicit = (
            rotated is True or angle is not None or coordinate_angle is not None
            or format in _ROTATED_FORMATS or len(coordinates) >= 5)
        if rotated is False and format not in _ROTATED_FORMATS and len(coordinates) == 4:
            self._rotation_explicit = False

        self._image_name = image_name
        self._type_coordinates = type_coordinates
        self._confidence = confidence
        self._class_id = class_id
        self._format = format
        if bb_type == BBType.DETECTED and confidence is None:
            raise IOError(
                'For bb_type=\'Detected\', it is necessary to inform the confidence value.')
        self._bb_type = bb_type

        if img_size is None:
            self._width_img = None
            self._height_img = None
        else:
            self._width_img = img_size[0]
            self._height_img = img_size[1]

        # If YOLO format (rel_x_center, rel_y_center, rel_width, rel_height), change it to absolute format (x,y,w,h)
        if format == BBFormat.YOLO:
            assert self._width_img is not None and self._height_img is not None
            if self._rotation_explicit:
                self._format = BBFormat.CXCYWHA
            else:
                self._format = BBFormat.XYWH
            self._type_coordinates = CoordinatesType.RELATIVE

        self.set_coordinates(coordinates,
                             img_size=img_size,
                             type_coordinates=self._type_coordinates,
                             angle=angle,
                             angle_unit=angle_unit,
                             angle_rad=angle_rad)

    def set_coordinates(self,
                        coordinates,
                        type_coordinates,
                        img_size=None,
                        angle=None,
                        angle_unit=None,
                        angle_rad=None):
        self._type_coordinates = type_coordinates
        if type_coordinates == CoordinatesType.RELATIVE and img_size is None:
            raise IOError(
                'Parameter \'img_size\' is required. It is necessary to inform the image size.')

        coordinates = _normalize_coordinates(coordinates)
        if angle is None and angle_rad is None and len(coordinates) >= 5:
            angle = coordinates[4]
        if angle is not None or angle_rad is not None:
            self._angle = normalize_angle_degrees(angle, angle_unit, angle_rad)
            self._rotation_explicit = True

        if getattr(self, '_rotation_explicit', False):
            if img_size is not None:
                self._width_img = img_size[0]
                self._height_img = img_size[1]
            cx, cy, w, h = map(float, coordinates[:4])
            if type_coordinates == CoordinatesType.RELATIVE:
                if img_size is None:
                    raise IOError(
                        'Parameter \'img_size\' is required. It is necessary to inform the image size.'
                    )
                width_img, height_img = img_size
                cx *= float(width_img)
                w *= float(width_img)
                cy *= float(height_img)
                h *= float(height_img)

            self._center_x = float(cx)
            self._center_y = float(cy)
            self._w = float(w)
            self._h = float(h)
            # Preserve the ordinary attributes as the unrotated extent.  These
            # are continuous open-ended coordinates for rotated/mixed IoU.
            self._x = self._center_x - self._w / 2.0
            self._y = self._center_y - self._h / 2.0
            self._x2 = self._center_x + self._w / 2.0
            self._y2 = self._center_y + self._h / 2.0
            if self._format not in _ROTATED_FORMATS:
                # Keep the user-facing format, but expose the center format for
                # callers that ask for the rotated representation.
                self._rotated_coordinate_format = BBFormat.CXCYWHA
            return

        # If relative coordinates, convert to absolute values
        # For relative coords: (x,y,w,h)=(X_center/img_width , Y_center/img_height)
        if (type_coordinates == CoordinatesType.RELATIVE):
            self._width_img = img_size[0]
            self._height_img = img_size[1]
            if self._format == BBFormat.XYWH:
                (self._x, self._y, self._w,
                 self._h) = convert_to_absolute_values(img_size, coordinates)
                self._x2 = self._w
                self._y2 = self._h
                self._w = self._x2 - self._x
                self._h = self._y2 - self._y
            elif self._format == BBFormat.XYX2Y2:
                x1, y1, x2, y2 = coordinates
                # Converting to absolute values
                self._x = round(x1 * self._width_img)
                self._x2 = round(x2 * self._width_img)
                self._y = round(y1 * self._height_img)
                self._y2 = round(y2 * self._height_img)
                self._w = self._x2 - self._x
                self._h = self._y2 - self._y
            else:
                raise IOError(
                    'For relative coordinates, the format must be XYWH (x,y,width,height)')
        # For absolute coords: (x,y,w,h)=real bb coords
        else:
            self._x = coordinates[0]
            self._y = coordinates[1]
            if self._format == BBFormat.XYWH:
                self._w = coordinates[2]
                self._h = coordinates[3]
                self._x2 = self._x + self._w
                self._y2 = self._y + self._h
            else:  # self._format == BBFormat.XYX2Y2: <left> <top> <right> <bottom>.
                self._x2 = coordinates[2]
                self._y2 = coordinates[3]
                self._w = self._x2 - self._x
                self._h = self._y2 - self._y
        # Convert all values to float
        self._x = float(self._x)
        self._y = float(self._y)
        self._w = float(self._w)
        self._h = float(self._h)
        self._x2 = float(self._x2)
        self._y2 = float(self._y2)

    def get_absolute_bounding_box(self, format=BBFormat.XYWH):
        """ Get bounding box in its absolute format.

        Parameters
        ----------
        format : Enum
            Format of the bounding box (BBFormat.XYWH or BBFormat.XYX2Y2) to be retreived.

        Returns
        -------
        tuple
            Four coordinates representing the absolute values of the bounding box.
            If specified format is BBFormat.XYWH, the coordinates are (upper-left-X, upper-left-Y,
            width, height).
            If format is BBFormat.XYX2Y2, the coordinates are (upper-left-X, upper-left-Y,
            bottom-right-X, bottom-right-Y).
        """
        if isinstance(format, str):
            format = BBFormat.parse(format)
        if format == BBFormat.XYWH:
            return (self._x, self._y, self._w, self._h)
        elif format == BBFormat.XYX2Y2:
            return (self._x, self._y, self._x2, self._y2)
        elif format == BBFormat.CXCYWH:
            return (self.get_center_x(), self.get_center_y(), self._w, self._h)
        elif format == BBFormat.CXCYWHA:
            return (self.get_center_x(), self.get_center_y(), self._w, self._h,
                    self.get_angle())
        elif format == BBFormat.CXCYWHA_RAD:
            return (self.get_center_x(), self.get_center_y(), self._w, self._h,
                    self.get_angle('radians'))

    def get_relative_bounding_box(self, img_size=None):
        """ Get bounding box in its relative format.

        Parameters
        ----------
        img_size : tuple
            Image size in the format (width, height)=>(int, int)

        Returns
        -------
        tuple
            Four coordinates representing the relative values of the bounding box (x,y,w,h) where:
                x,y : bounding_box_center/width_of_the_image
                w   : bounding_box_width/width_of_the_image
                h   : bounding_box_height/height_of_the_image
        """
        if img_size is None and self._width_img is None and self._height_img is None:
            raise IOError(
                'Parameter \'img_size\' is required. It is necessary to inform the image size.')
        if img_size is not None:
            return convert_to_relative_values((img_size[0], img_size[1]),
                                              (self._x, self._x2, self._y, self._y2))
        else:
            return convert_to_relative_values((self._width_img, self._height_img),
                                              (self._x, self._x2, self._y, self._y2))

    def get_image_name(self):
        """ Get the string that represents the image.

        Returns
        -------
        string
            Name of the image.
        """
        return self._image_name

    def get_confidence(self):
        """ Get the confidence level of the detection. If bounding box type is BBType.GROUND_TRUTH,
        the confidence is None.

        Returns
        -------
        float
            Value between 0 and 1 representing the confidence of the detection.
        """
        return self._confidence

    def get_format(self):
        """ Get the format of the bounding box (BBFormat.XYWH or BBFormat.XYX2Y2).

        Returns
        -------
        Enum
            Format of the bounding box. It can be either:
                BBFormat.XYWH: <left> <top> <width> <height>
                BBFomat.XYX2Y2: <left> <top> <right> <bottom>.
        """
        return self._format

    def set_class_id(self, class_id):
        self._class_id = class_id

    def set_bb_type(self, bb_type):
        self._bb_type = bb_type

    def get_class_id(self):
        """ Get the class of the object the bounding box represents.

        Returns
        -------
        string
            Class of the detected object (e.g. 'cat', 'dog', 'person', etc)
        """
        return self._class_id

    def get_image_size(self):
        """ Get the size of the image where the bounding box is represented.

        Returns
        -------
        tupe
            Image size in pixels in the format (width, height)=>(int, int)
        """
        return (self._width_img, self._height_img)

    def get_center_x(self):
        if hasattr(self, '_center_x'):
            return self._center_x
        return (self._x + self._x2) / 2.0

    def get_center_y(self):
        if hasattr(self, '_center_y'):
            return self._center_y
        return (self._y + self._y2) / 2.0

    def get_center(self):
        return (self.get_center_x(), self.get_center_y())

    def get_angle(self, unit='degrees'):
        angle = getattr(self, '_angle', 0.0)
        unit = 'degrees' if unit is None else str(unit).lower()
        if unit in ('rad', 'radian', 'radians'):
            return math.radians(angle)
        return angle

    def is_rotated(self):
        """Whether this box uses continuous rotated-polygon geometry."""
        return bool(getattr(self, '_rotation_explicit', False))

    def set_angle(self, angle, unit=None, angle_rad=None):
        self._angle = normalize_angle_degrees(angle, unit, angle_rad)
        self._rotation_explicit = True
        return self._angle

    @property
    def angle(self):
        return self.get_angle()

    def get_rotated_coordinates(self, unit='degrees'):
        return (self.get_center_x(), self.get_center_y(), self._w, self._h,
                self.get_angle(unit))

    def get_continuous_area(self):
        """Open-ended geometric area, including for a rotated rectangle."""
        if self.is_rotated():
            return abs(float(self._w) * float(self._h))
        return abs(float(self._x2 - self._x) * float(self._y2 - self._y))

    def get_continuous_polygon(self):
        """Return four continuous corner coordinates, ordered around the box."""
        if self.is_rotated():
            return rotated_corners(self.get_center_x(), self.get_center_y(),
                                   self._w, self._h, self.get_angle())
        x1, y1, x2, y2 = self._x, self._y, self._x2, self._y2
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    def get_area(self):
        if self.is_rotated():
            # Rotated evaluation is continuous geometry: a 4x1 box is 4 and
            # does not receive the legacy raster "+1" border convention.
            return self.get_continuous_area()
        assert isclose(self._w * self._h, (self._x2 - self._x) * (self._y2 - self._y))
        assert (self._x2 >= self._x)
        assert (self._y2 >= self._y)
        return (self._x2 - self._x + 1) * (self._y2 - self._y + 1)

    def get_coordinates_type(self):
        """ Get type of the coordinates (CoordinatesType.RELATIVE or CoordinatesType.ABSOLUTE).

        Returns
        -------
        Enum
            Enum representing if the bounding box coordinates (x,y,w,h) are absolute or relative
                to size of the image (CoordinatesType.RELATIVE or CoordinatesType.ABSOLUTE).
        """
        return self._type_coordinates

    def get_bb_type(self):
        """ Get type of the bounding box that represents if it is a ground-truth or detected box.

        Returns
        -------
        Enum
            Enum representing the type of the bounding box (BBType.GROUND_TRUTH or BBType.DETECTED)
        """
        return self._bb_type

    def __str__(self):
        abs_bb_xywh = self.get_absolute_bounding_box(format=BBFormat.XYWH)
        abs_bb_xyx2y2 = self.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
        area = self.get_area()
        return f'image name: {self._image_name}\nclass: {self._class_id}\nbb (XYWH): {abs_bb_xywh}\nbb (X1Y1X2Y2): {abs_bb_xyx2y2}\narea: {area}\nbb_type: {self._bb_type}'

    def __eq__(self, other):
        if not isinstance(other, BoundingBox):
            # unrelated types
            return False
        return str(self) == str(other)

    @staticmethod
    def compare(det1, det2):
        """ Static function to compare if two bounding boxes represent the same area in the image,
            regardless the format of their boxes.

        Parameters
        ----------
        det1 : BoundingBox
            BoundingBox object representing one bounding box.
        dete2 : BoundingBox
            BoundingBox object representing another bounding box.

        Returns
        -------
        bool
            True if both bounding boxes have the same coordinates, otherwise False.
        """
        det1BB = det1.getAbsoluteBoundingBox()
        det1img_size = det1.getImageSize()
        det2BB = det2.getAbsoluteBoundingBox()
        det2img_size = det2.getImageSize()

        if det1.get_class_id() == det2.get_class_id() and \
           det1.get_confidence() == det2.get_confidence() and \
           det1BB[0] == det2BB[0] and \
           det1BB[1] == det2BB[1] and \
           det1BB[2] == det2BB[2] and \
           det1BB[3] == det2BB[3] and \
           det1img_size[0] == det1img_size[0] and \
           det2img_size[1] == det2img_size[1]:
            return True
        return False

    @staticmethod
    def clone(bounding_box):
        """ Static function to clone a given bounding box.

        Parameters
        ----------
        bounding_box : BoundingBox
            Bounding box object to be cloned.

        Returns
        -------
        BoundingBox
            Cloned BoundingBox object.
        """
        absBB = bounding_box.get_absolute_bounding_box(format=BBFormat.XYWH)
        # return (self._x,self._y,self._x2,self._y2)
        new_bounding_box = BoundingBox(bounding_box.get_image_name(),
                                       bounding_box.get_class_id(),
                                       absBB[0],
                                       absBB[1],
                                       absBB[2],
                                       absBB[3],
                                       type_coordinates=bounding_box.getCoordinatesType(),
                                       img_size=bounding_box.getImageSize(),
                                       bb_type=bounding_box.getbb_type(),
                                       confidence=bounding_box.getConfidence(),
                                       format=BBFormat.XYWH)
        return new_bounding_box

    @staticmethod
    def continuous_iou(boxA, boxB):
        """IoU using continuous polygon geometry for box pairs."""
        return geometry_iou(boxA, boxB)

    @staticmethod
    def rotated_iou(boxA, boxB):
        """Alias for :meth:`continuous_iou`."""
        return geometry_iou(boxA, boxB)

    @staticmethod
    def polygon_iou(boxA, boxB):
        """Alias for :meth:`continuous_iou`."""
        return geometry_iou(boxA, boxB)

    @staticmethod
    def iou(boxA, boxB, continuous=False, rotated=None, **kwargs):
        # Rotated boxes, including rotated/axis-aligned mixed pairs, use their
        # real continuous polygons.  ``continuous``/``rotated=True`` explicitly
        # requests polygon geometry for ordinary boxes as well.  Ordinary
        # BoundingBox pairs otherwise retain the original inclusive convention.
        continuous = continuous or any(
            kwargs.get(name) for name in
            ('use_continuous', 'continuous_area', 'polygon', 'use_polygon'))
        rotated_a = isinstance(boxA, BoundingBox) and boxA.is_rotated()
        rotated_b = isinstance(boxB, BoundingBox) and boxB.is_rotated()
        raw_rotated_a = not isinstance(boxA, BoundingBox) and len(boxA) == 5
        raw_rotated_b = not isinstance(boxB, BoundingBox) and len(boxB) == 5
        if continuous or rotated is True or rotated_a or rotated_b or raw_rotated_a or raw_rotated_b:
            return geometry_iou(boxA, boxB)

        coords_A = boxA.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
        coords_B = boxB.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
        # if boxes do not intersect
        if BoundingBox.have_intersection(coords_A, coords_B) is False:
            return 0
        interArea = BoundingBox.get_intersection_area(coords_A, coords_B)
        union = BoundingBox.get_union_areas(boxA, boxB, interArea=interArea)
        # intersection over union
        iou = interArea / union
        assert iou >= 0
        return iou

    # boxA = (Ax1,Ay1,Ax2,Ay2)
    # boxB = (Bx1,By1,Bx2,By2)
    @staticmethod
    def have_intersection(boxA, boxB):
        if isinstance(boxA, BoundingBox):
            boxA = boxA.get_absolute_bounding_box(BBFormat.XYX2Y2)
        if isinstance(boxB, BoundingBox):
            boxB = boxB.get_absolute_bounding_box(BBFormat.XYX2Y2)
        if boxA[0] > boxB[2]:
            return False  # boxA is right of boxB
        if boxB[0] > boxA[2]:
            return False  # boxA is left of boxB
        if boxA[3] < boxB[1]:
            return False  # boxA is above boxB
        if boxA[1] > boxB[3]:
            return False  # boxA is below boxB
        return True

    @staticmethod
    def get_intersection_area(boxA, boxB):
        rotated_a = isinstance(boxA, BoundingBox) and boxA.is_rotated()
        rotated_b = isinstance(boxB, BoundingBox) and boxB.is_rotated()
        if rotated_a or rotated_b:
            return polygon_intersection_area(boxA.get_continuous_polygon(),
                                             boxB.get_continuous_polygon())
        if isinstance(boxA, BoundingBox):
            boxA = boxA.get_absolute_bounding_box(BBFormat.XYX2Y2)
        if isinstance(boxB, BoundingBox):
            boxB = boxB.get_absolute_bounding_box(BBFormat.XYX2Y2)
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # intersection area
        return (xB - xA + 1) * (yB - yA + 1)

    @staticmethod
    def get_union_areas(boxA, boxB, interArea=None):
        if ((isinstance(boxA, BoundingBox) and boxA.is_rotated())
                or (isinstance(boxB, BoundingBox) and boxB.is_rotated())):
            # Mixed rotated evaluation uses continuous geometry on both sides.
            area_A = boxA.get_continuous_area()
            area_B = boxB.get_continuous_area()
        else:
            area_A = boxA.get_area()
            area_B = boxB.get_area()
        if interArea is None:
            interArea = BoundingBox.get_intersection_area(boxA, boxB)
        return float(area_A + area_B - interArea)

    @staticmethod
    def get_amount_bounding_box_all_classes(bounding_boxes, reverse=False):
        classes = list(set([bb._class_id for bb in bounding_boxes]))
        ret = {}
        for c in classes:
            ret[c] = len(BoundingBox.get_bounding_box_by_class(bounding_boxes, c))
        # Sort dictionary by the amount of bounding boxes
        ret = {k: v for k, v in sorted(ret.items(), key=lambda item: item[1], reverse=reverse)}
        return ret

    @staticmethod
    def get_bounding_box_by_class(bounding_boxes, class_id):
        # get only specified bounding box type
        return [bb for bb in bounding_boxes if bb.get_class_id() == class_id]

    @staticmethod
    def get_bounding_boxes_by_image_name(bounding_boxes, image_name):
        # get only specified bounding box type
        return [bb for bb in bounding_boxes if bb.get_image_name() == image_name]

    @staticmethod
    def get_total_images(bounding_boxes):
        return len(list(set([bb.get_image_name() for bb in bounding_boxes])))

    @staticmethod
    def get_average_area(bounding_boxes):
        areas = [bb.get_area() for bb in bounding_boxes]
        return sum(areas) / len(areas)


class RotatedBoundingBox(BoundingBox):
    """Convenience BoundingBox whose coordinates are ``(cx, cy, w, h, angle)``.

    It accepts either the usual BoundingBox positional style with a 5-element
    ``coordinates`` value or explicit ``cx, cy, width, height, angle`` values.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('format', BBFormat.CXCYWHA)

        coordinates = kwargs.get('coordinates')
        cx = kwargs.pop('cx', None)
        cy = kwargs.pop('cy', None)
        width = kwargs.pop('width', kwargs.pop('w', None))
        height = kwargs.pop('height', kwargs.pop('h', None))
        angle = kwargs.pop('angle', 0.0)

        if coordinates is None and None not in (cx, cy, width, height):
            kwargs['coordinates'] = (cx, cy, width, height, angle)
        elif coordinates is None and len(args) >= 7:
            # Positional form:
            # image, class, cx, cy, width, height, angle, [remaining BoundingBox args]
            maybe_coords = args[2:7]
            if all(isinstance(v, (int, float)) for v in maybe_coords):
                args = args[:2] + (tuple(maybe_coords),) + args[7:]

        super().__init__(*args, **kwargs)


# Descriptive aliases for downstream code and tests.
RotationBoundingBox = RotatedBoundingBox
RotatingBoundingBox = RotatedBoundingBox
RotatedBBox = RotatedBoundingBox
RotatedBB = RotatedBoundingBox
RotatedBox = RotatedBoundingBox
RotationBBox = RotatedBoundingBox
RotationBB = RotatedBoundingBox
RotationBox = RotatedBoundingBox
RBox = RotatedBoundingBox
OBBBoundingBox = RotatedBoundingBox
OBBBox = RotatedBoundingBox
OBB = RotatedBoundingBox
obb_iou = BoundingBox.continuous_iou
rotated_iou = BoundingBox.continuous_iou
polygon_iou = BoundingBox.continuous_iou
iou = BoundingBox.iou
calculate_iou = BoundingBox.iou
get_iou = BoundingBox.iou
