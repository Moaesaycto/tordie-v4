from numpy import sin, cos, sqrt, arctan2
from svgwrite.drawing import Drawing
from shapes.shape import Shape
from copy import deepcopy
import numpy as np

from options import *

class Point(Shape):
    def __init__(self, coord1, coord2, **kwargs):
        """
        Assumes Cartesian unless 'polar=True', where coord1 is r and coord2 is theta
        """
        super().__init__(**kwargs)
        if kwargs.get("polar", False):
            self.x, self.y = coord1*cos(coord2), coord1*sin(coord2)
            self.r, self.theta = coord1, coord2
        else:
            self.x, self.y = coord1, coord2
            self.r, self.theta = sqrt(coord1**2 + coord2**2), arctan2(self.y, self.x)

    def __add__(self, other): return Point(self.x + other.x, self.y + other.y)
    def __sub__(self, other): return Point(self.x - other.x, self.y - other.y)
    def __pow__(self, other):
        arg, mod = self.theta*other, self.r**other
        return Point(mod*np.cos(arg), mod*np.sin(arg))
    def __mul__(self, other):
        if isinstance(other, Point): return Point(self.x*other.x - self.y*other.y, self.x*other.y + self.y*other.x)
        if isinstance(other, (int, float)): return Point(self.x * other, self.y * other) # Scalar multiplication
    def __truediv__(self, other):
        if isinstance(other, Point): return Point(self.x*other.x + self.y*other.y, -self.x*other.y + self.y*other.x) / (other.x**2 + other.y**2)
        if isinstance(other, (int, float)): return Point(self.x / other, self.y / other)
    def __abs__(self):
        return self.r
    def set_cartesian(self, x, y):
        self.x, self.y = x, y
        self.r, self.theta = sqrt(x**2 + y**2), arctan2(y, x)
    
    def set_polar(self, r, theta):
        self.x, self.y = r*cos(theta), r*sin(theta)
        self.r, self.theta = r, theta

    def adjust(self, width, height, scale) -> tuple:
        return (
            self.x / 2* float(width / scale[0]) + (width + BOUNDS_FIX) / 2,
            -self.y / 2 * float(height / SCALE[1]) + (height + BOUNDS_FIX)/ 2
        )

    def draw(self, svg: Drawing):
        disk = svg.drawing.add(svg.drawing.circle(self.adjust(svg.width, svg.height, SCALE), self.width))
        disk.fill(self.stroke).stroke(self.stroke, width=0)

    def reflect(self, point):
        new_point = deepcopy(point)
        new_point.set_cartesian(2*self.x - point.x, 2*self.y - point.y)
        return new_point
