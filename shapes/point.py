from numpy import sin, cos, sqrt, arctan2
from svgwrite.drawing import Drawing

from options import *

class Point:
    def __init__(self, coord1, coord2, **kwargs):
        """
        Assumes Cartesian unless 'polar=True', where coord1 is r and coord2 is theta
        """
        if kwargs.get("polar", False):
            self.x, self.y = coord1*cos(coord2), coord1*sin(coord2)
            self.r, self.theta = coord1, coord2
        else:
            self.x, self.y = coord1, coord2
            self.r, self.theta = sqrt(coord1**2 + coord2**2), arctan2(self.y, self.x)
        
        self.radius = kwargs.get("radius", DEFAULT_RADIUS)
        self.stroke = kwargs.get("stroke", DEFAULT_STROKE)

    def adjust(self, width, height, scale) -> tuple:
        return (
            self.x / 2* float(width / scale[0]) + (width + BOUNDS_FIX) / 2,
            -self.y / 2 * float(height / SCALE[1]) + (height + BOUNDS_FIX)/ 2
        )

    def draw(self, svg: Drawing):
        disk = svg.drawing.add(svg.drawing.circle(self.adjust(svg.width, svg.height, SCALE), self.radius))
        disk.fill(self.stroke).stroke(self.stroke, width=0)
