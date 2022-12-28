from svgwrite.drawing import Drawing
from shapes.shape import Shape
from copy import deepcopy

from options import *

class Circle(Shape):
    def __init__(self, center, radius, **kwargs):
        super().__init__(**kwargs)
        self.center = center
        self.radius = radius
    
    def draw(self, svg):
        disk = svg.drawing.add(svg.drawing.circle(self.center.adjust(svg.width, svg.height, SCALE), self.radius / 2 * float(svg.width / SCALE[0])))
        disk.fill('white', opacity=0).stroke(self.stroke, width=self.width)
    
    def reflect(self, point):
        new_point = deepcopy(point)
        R = self.radius
        h = self.center.x
        k = self.center.y

        x0 = point.x
        y0 = point.y

        alpha = R**2/((x0-h)**2+(y0-k)**2)
        new_point.set_cartesian(alpha*(x0-h)+h, alpha*(y0-k)+k)
        return new_point
