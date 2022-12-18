from abc import ABC, abstractmethod
from svgwrite.drawing import Drawing
from svgwrite import shapes
from options import *

class Line(ABC):
    def __init__(self, point1, point2, **kwargs):
        self.thickness = kwargs.get("thickness", DEFAULT_THICKNESS)
        self.stroke = kwargs.get("stroke", DEFAULT_STROKE)

        self.start, self.end = point1, point2

    @abstractmethod
    def draw(): pass

    """ @abstractmethod
    def reflect(point): pass """


class EuclideanLine(Line):
    def __init__(self, point1, point2, **kwargs):
        super().__init__(point1, point2, **kwargs)

    def draw(self, svg: Drawing):
        svg.drawing.add(shapes.Line(start=self.start.adjust(svg.width, svg.height, SCALE),
                            end=self.end.adjust(svg.width, svg.height, SCALE),
                            stroke_width=self.thickness,stroke=self.stroke))