from abc import ABC, abstractmethod
from svgwrite.drawing import Drawing
from svgwrite import shapes
from settings import *
from shapes.point import Point
from copy import deepcopy
from utils import color
from shapes.shape import Shape
from shapes.circle import Circle
from numpy import sqrt


class Line(Shape):
    def __init__(self, point1, point2, **kwargs):
        self.thickness = kwargs.get("thickness", DEFAULT_THICKNESS)
        self.stroke = kwargs.get("stroke", DEFAULT_STROKE)

        self.start, self.end = point1, point2

    @abstractmethod
    def reflect(self, point: Point): pass
    
    @abstractmethod
    def conformal(self, f): pass

    def translate(self, vector):
        self.start.translate(vector)
        self.end.translate(vector)

    def rotate(self, angle, *args): 
        self.start.rotate(angle, *args)
        self.end.rotate(angle, *args)

    def scale(self, scale, *args):
        self.start.scale(scale, *args)
        self.end.scale(scale, *args)


class EuclideanLine(Line):
    def __init__(self, point1, point2, **kwargs):
        super().__init__(point1, point2, **kwargs)

    def draw(self, svg: Drawing):
        svg.drawing.add(shapes.Line(start=self.start.adjust(svg.width, svg.height, SCALE),
                            end=self.end.adjust(svg.width, svg.height, SCALE),
                            stroke_width=self.thickness, stroke=self.stroke))
    
    def reflect(self, point):
        new_point = deepcopy(point)
        x0, y0, x1, y1 = self.start.x, self.start.y, self.end.x, self.end.y
        if x0 == x1:
            new_point.set_cartesian(2*self.start.x - point.x, point.y)
        elif y0 == y1:
             new_point.set_cartesian(point.x, 2*self.start.y - point.y)
        else:
            m = (y1 - y0)/(x1 - x0)
            r = (-m*point.y - point.x + m*y0 - m**2*x0)/(-1-m**2)
            new_point.set_cartesian(2*r-point.x, 2*(-1/m*(r - point.x)) + point.y)
            
        return new_point
    
    def conformal(self, f):
        result = deepcopy(self)
        result.start = f(self.start)
        result.end = f(self.end)
        return result


class PoincareLine(Line):
    def __init__(self, point1, point2, **kwargs):
        if point1.r > 1 or point2.r > 1: raise ValueError(color.red + "Point cannot be outside of disk." + color.end)
        super().__init__(point1, point2, **kwargs)
        self.circle = self.arc()


    def arc(self):
            point0, point1 = self.start, self.end
            if point0.r*point1.r == 0 or (point0.y/point0.r == -point1.y/point1.r and point0.x/point0.r == -point1.x/point1.r): return Line(point0, point1)

            # FOR INSTANCE N = 7 FOR POINCARE TESSELLATION
            #if abs(point1.y) <= POINCARE_ERR: point0, point1 = point1, point0

            # FOR INSTANCE N = 8 FOR POINCARE TESSELLATION
            if abs(point1.y) <= POINCARE_ERR and point1.x > 0: point0, point1 = point1, point0
            elif not (abs(point1.y) <= POINCARE_ERR and point1.x < 0):
                point0, point1 = point1, point0
                if abs(point1.y) <= POINCARE_ERR and point1.x > 0: point0, point1 = point1, point0

            disk = Circle(Point(0,0), 1)
            invpoint0, invpoint1 = disk.reflect(point0), disk.reflect(point1)

            x11, y11 = point0.x, point0.y
            x12, y12 = invpoint0.x, invpoint0.y

            if abs(y12 - y11) <= POINCARE_ERR: m1 = -1/MAN_EPS
            else: m1 = -(x12 - x11)/(y12-y11)

            b1 = (y12 + y11)/2 - m1*(x12 + x11)/2

            x21, y21 = point1.x, point1.y
            x22, y22 = invpoint1.x, invpoint1.y

            if abs(y22 - y21) <= POINCARE_ERR: m2 = 0
            else: m2 = -(x22 - x21)/(y22 - y21)

            b2 = (y22 + y21)/2 - m2*(x22 + x21)/2

            center = Point((b2 - b1)/(m1-m2), m1*(b2 - b1)/(m1 - m2)+b1)
            return Circle(center, sqrt((center.x - point1.x)**2 + (center.y - point1.y)**2))


    def draw(self, svg: Drawing):
        svg.draw(self.circle)
    

    def reflect(self, point):
        return self.circle.reflect(point)

    def conformal(self, f):
        pass