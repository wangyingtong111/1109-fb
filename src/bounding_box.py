from math import isclose

from src.utils.general_utils import (clip_to_image_size, convert_to_absolute_values,
                                     convert_to_relative_values)

from .utils.enumerators import BBFormat, BBType, CoordinatesPrecision, CoordinatesType


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
                 precision=CoordinatesPrecision.QUANTIZED):
        """ Constructor.

        Parameters
        ----------
            image_name : str
                String representing the name of the image.
            class_id : str
                String value representing class id.
            coordinates : tuple
                Tuple with 4 elements whose values (float) represent coordinates of the bounding \\
                    box.
                The coordinates can be (x, y, w, h)=>(float,float,float,float) or(x1, y1, x2, y2)\\
                    =>(float,float,float,float).
                See parameter `format`.
            type_coordinates : Enum (optional)
                Enum representing if the bounding box coordinates (x,y,w,h) are absolute or relative
                to size of the image. Default:'Absolute'.
            img_size : tuple (optional)
                Image size in the format (width, height)=>(int, int) representinh the size of the
                image of the bounding box. If type_coordinates is 'Relative', img_size is required.
            bb_type : Enum (optional)
                Enum identifying if the bounding box is a ground truth or a detection. If it is a
                detection, the confidence must be informed.
            confidence : float (optional)
                Value representing the confidence of the detected object. If detectionType is
                Detection, confidence needs to be informed.
            format : Enum
                Enum (BBFormat.XYWH or BBFormat.XYX2Y2) indicating the format of the coordinates of
                the bounding boxes.
                BBFormat.XYWH: <left> <top> <width> <height>
                BBFomat.XYX2Y2: <left> <top> <right> <bottom>.
                BBFomat.YOLO: <x_center> <y_center> <width> <height>. (relative)
            precision : Enum (optional)
                Enum (CoordinatesPrecision.EXACT or CoordinatesPrecision.QUANTIZED) indicating how
                the relative coordinates are converted to absolute values.
                CoordinatesPrecision.EXACT keeps the sub-pixel float borders, so the value informed
                in relative coordinates round-trips back to the same relative value. On a 100x50
                image, center x=.5 and relative width .005 give the x borders 49.75 and 50.25.
                CoordinatesPrecision.QUANTIZED keeps the legacy behaviour, rounding the borders to
                whole pixels. Default: CoordinatesPrecision.QUANTIZED.
        """

        self._image_name = image_name
        self._type_coordinates = type_coordinates
        self._confidence = confidence
        self._class_id = class_id
        self._format = format
        self.set_precision(precision)
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
            self._format = BBFormat.XYWH
            self._type_coordinates = CoordinatesType.RELATIVE

        self.set_coordinates(coordinates,
                             img_size=img_size,
                             type_coordinates=self._type_coordinates)

    def set_precision(self, precision):
        """ Set how relative coordinates are converted into absolute ones.

        Parameters
        ----------
        precision : Enum
            CoordinatesPrecision.EXACT keeps the sub-pixel float borders of relative boxes,
            CoordinatesPrecision.QUANTIZED keeps the legacy whole-pixel rounding. The precision
            never changes boxes that were informed in absolute coordinates.
        """
        if precision is None:
            # boxes created without an explicit precision behave as the legacy ones
            precision = CoordinatesPrecision.QUANTIZED
        if not isinstance(precision, CoordinatesPrecision):
            raise IOError(
                "Parameter 'precision' must be a member of the CoordinatesPrecision enumerator "
                "(CoordinatesPrecision.EXACT or CoordinatesPrecision.QUANTIZED).")
        self._precision = precision

    def get_precision(self):
        """ Get how relative coordinates are converted into absolute ones.

        Returns
        -------
        Enum
            CoordinatesPrecision.EXACT or CoordinatesPrecision.QUANTIZED.
        """
        return self._precision

    def set_coordinates(self, coordinates, type_coordinates, img_size=None):
        self._type_coordinates = type_coordinates
        # Normalized values informed by the caller, kept so that the relative view round-trips
        # without any loss. It is only used when the coordinates were informed as relative and
        # the exact precision is selected.
        self._relative_coordinates = None
        if type_coordinates == CoordinatesType.RELATIVE and img_size is None:
            raise IOError(
                'Parameter \'img_size\' is required. It is necessary to inform the image size.')

        # If relative coordinates, convert to absolute values
        # For relative coords: (x,y,w,h)=(X_center/img_width , Y_center/img_height)
        if (type_coordinates == CoordinatesType.RELATIVE):
            self._width_img = img_size[0]
            self._height_img = img_size[1]
            if self._format == BBFormat.XYWH:
                (self._x, self._y, self._x2,
                 self._y2) = convert_to_absolute_values(img_size,
                                                       coordinates,
                                                       precision=self._precision)
                self._w = self._x2 - self._x
                self._h = self._y2 - self._y
                if self._precision == CoordinatesPrecision.EXACT:
                    self._relative_coordinates = (float(coordinates[0]), float(coordinates[1]),
                                                  float(coordinates[2]), float(coordinates[3]))
            elif self._format == BBFormat.XYX2Y2:
                x1, y1, x2, y2 = coordinates
                # Converting to absolute values
                if self._precision == CoordinatesPrecision.EXACT:
                    self._x = float(x1) * self._width_img
                    self._x2 = float(x2) * self._width_img
                    self._y = float(y1) * self._height_img
                    self._y2 = float(y2) * self._height_img
                    self._relative_coordinates = (
                        (float(x1) + float(x2)) / 2.0,
                        (float(y1) + float(y2)) / 2.0,
                        float(x2) - float(x1),
                        float(y2) - float(y1),
                    )
                else:
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

    def _get_image_size_or_raise(self, img_size=None):
        """ Resolve the image size to be used by the clipped views.

        Parameters
        ----------
        img_size : tuple (optional)
            Image size (width, height). If None, the size informed when the box was created is
            used.

        Returns
        -------
        tuple
            Image size (width, height). The size informed when the box was created is returned
            unchanged so that the clipped view keeps the original type.
        """
        if img_size is not None:
            return (img_size[0], img_size[1])
        if self._width_img is None or self._height_img is None:
            raise IOError(
                'Parameter \'img_size\' is required. It is necessary to inform the image size.')
        return (self._width_img, self._height_img)

    def get_clipped_bounding_box(self, img_size=None):
        """ Get a new bounding box limited to the visible area of the image.

        The clipping is a *view*: the current box is never modified, so its
        (unclipped) absolute and relative coordinates stay untouched no matter
        how many times a clipped view is read.

        Parameters
        ----------
        img_size : tuple (optional)
            Image size in the format (width, height)=>(int, int). If None, the size informed when
            the box was created is used.

        Returns
        -------
        BoundingBox
            A new box whose coordinates are limited to [0, width] x [0, height]. It keeps the
            image name, class, confidence, format, type and precision of the current box.
            Its relative view describes the visible geometry, e.g. a box with center x=0 and
            relative width .01 on a 100 pixel wide image has a clipped width of .005.
        """
        size = self._get_image_size_or_raise(img_size)
        x1, y1, x2, y2 = clip_to_image_size((self._x, self._y, self._x2, self._y2), size)
        return BoundingBox(image_name=self._image_name,
                           class_id=self._class_id,
                           coordinates=(x1, y1, x2, y2),
                           type_coordinates=CoordinatesType.ABSOLUTE,
                           img_size=size,
                           bb_type=self._bb_type,
                           confidence=self._confidence,
                           format=BBFormat.XYX2Y2,
                           precision=self._precision)

    def get_absolute_bounding_box(self, format=BBFormat.XYWH, clip=False, img_size=None):
        """ Get bounding box in its absolute format.

        Parameters
        ----------
        format : Enum
            Format of the bounding box (BBFormat.XYWH or BBFormat.XYX2Y2) to be retreived.
        clip : bool (optional)
            If True, the returned coordinates are limited to the visible area of the image. The
            box itself is not modified. Default: False.
        img_size : tuple (optional)
            Image size (width, height) used when clip is True. If None, the size informed when the
            box was created is used.

        Returns
        -------
        tuple
            Four coordinates representing the absolute values of the bounding box.
            If specified format is BBFormat.XYWH, the coordinates are (upper-left-X, upper-left-Y,
            width, height).
            If format is BBFormat.XYX2Y2, the coordinates are (upper-left-X, upper-left-Y,
            bottom-right-X, bottom-right-Y).
        """
        if clip:
            return self.get_clipped_bounding_box(img_size).get_absolute_bounding_box(format)
        if format == BBFormat.XYWH:
            return (self._x, self._y, self._w, self._h)
        elif format == BBFormat.XYX2Y2:
            return (self._x, self._y, self._x2, self._y2)

    def get_relative_bounding_box(self, img_size=None, clip=False):
        """ Get bounding box in its relative format.

        Parameters
        ----------
        img_size : tuple
            Image size in the format (width, height)=>(int, int)
        clip : bool (optional)
            If True, the relative values describe the clipped (visible) geometry instead of the
            stored geometry. The box itself is not modified. Default: False.

        Returns
        -------
        tuple
            Four coordinates representing the relative values of the bounding box (x,y,w,h) where:
                x,y : bounding_box_center/width_of_the_image
                w   : bounding_box_width/width_of_the_image
                h   : bounding_box_height/height_of_the_image
        """
        if clip:
            return self.get_clipped_bounding_box(img_size).get_relative_bounding_box()
        if img_size is None and self._width_img is None and self._height_img is None:
            raise IOError(
                'Parameter \'img_size\' is required. It is necessary to inform the image size.')
        if img_size is not None:
            return convert_to_relative_values((img_size[0], img_size[1]),
                                              (self._x, self._x2, self._y, self._y2))
        # For relative boxes converted with exact precision, the informed normalized values are
        # returned as they are, so the round trip cannot lose sub-pixel information.
        if self._relative_coordinates is not None:
            return self._relative_coordinates
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

    def get_area(self, continuous=False, clip=False, img_size=None):
        """ Get the area of the bounding box.

        Parameters
        ----------
        continuous : bool (optional)
            If True, the area is computed from the continuous float borders
            (width x height) instead of the legacy inclusive whole-pixel rule. Use it together
            with the sub-pixel coordinates so that small boxes keep a non-zero area.
            Default: False (legacy inclusive rule).
        clip : bool (optional)
            If True, the area of the clipped (visible) box is returned. The box is not modified.
            Default: False.
        img_size : tuple (optional)
            Image size (width, height) used when clip is True.

        Returns
        -------
        float
            Area of the bounding box.
        """
        if clip:
            return self.get_clipped_bounding_box(img_size).get_area(continuous=continuous)
        if continuous:
            return max(self._x2 - self._x, 0.0) * max(self._y2 - self._y, 0.0)
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
        det1BB = det1.get_absolute_bounding_box()
        det1img_size = det1.get_image_size()
        det2BB = det2.get_absolute_bounding_box()
        det2img_size = det2.get_image_size()

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
                                       coordinates=absBB,
                                       type_coordinates=bounding_box.get_coordinates_type(),
                                       img_size=bounding_box.get_image_size(),
                                       bb_type=bounding_box.get_bb_type(),
                                       confidence=bounding_box.get_confidence(),
                                       format=BBFormat.XYWH,
                                       precision=bounding_box.get_precision())
        return new_bounding_box

    @staticmethod
    def iou(boxA, boxB, continuous=False, clip=False, img_size=None):
        """ Intersection over union of two bounding boxes.

        Parameters
        ----------
        boxA, boxB : BoundingBox
            Boxes to be compared.
        continuous : bool (optional)
            If True, the comparison uses the continuous float borders (half-open intervals) so
            that sub-pixel boxes keep their exact geometry and boxes that only touch at an edge
            share no area. If False (default), the legacy inclusive whole-pixel rule is used.
        clip : bool (optional)
            If True, both boxes are first limited to the visible area of the image. This is a
            view: the received boxes (and their unclipped relative values) are never modified.
            Default: False.
        img_size : tuple (optional)
            Image size (width, height) used when clip is True. If None, the size informed when
            each box was created is used.

        Returns
        -------
        float
            Intersection over union in the interval [0, 1].
        """
        if clip:
            boxA = boxA.get_clipped_bounding_box(img_size)
            boxB = boxB.get_clipped_bounding_box(img_size)
        coords_A = boxA.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
        coords_B = boxB.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
        if continuous:
            # Half-open intervals: only an edge contact (zero width/height) is not an overlap.
            if coords_A[0] >= coords_B[2] or coords_B[0] >= coords_A[2]:
                return 0.0
            if coords_A[3] <= coords_B[1] or coords_B[3] <= coords_A[1]:
                return 0.0
            interArea = BoundingBox.get_intersection_area(coords_A,
                                                          coords_B,
                                                          continuous=True)
            area_A = boxA.get_area(continuous=True)
            area_B = boxB.get_area(continuous=True)
            union = area_A + area_B - interArea
            if union <= 0:
                return 0.0
            return interArea / union
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
    def get_intersection_area(boxA, boxB, continuous=False):
        if isinstance(boxA, BoundingBox):
            boxA = boxA.get_absolute_bounding_box(BBFormat.XYX2Y2)
        if isinstance(boxB, BoundingBox):
            boxB = boxB.get_absolute_bounding_box(BBFormat.XYX2Y2)
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        if continuous:
            # Half-open intervals: a mere edge contact has zero width/height.
            return max(xB - xA, 0.0) * max(yB - yA, 0.0)
        # intersection area
        return (xB - xA + 1) * (yB - yA + 1)

    @staticmethod
    def get_union_areas(boxA, boxB, interArea=None, continuous=False):
        area_A = boxA.get_area(continuous=continuous)
        area_B = boxB.get_area(continuous=continuous)
        if interArea is None:
            interArea = BoundingBox.get_intersection_area(boxA, boxB, continuous=continuous)
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
    def get_average_area(bounding_boxes, continuous=False):
        areas = [bb.get_area(continuous=continuous) for bb in bounding_boxes]
        return sum(areas) / len(areas)
