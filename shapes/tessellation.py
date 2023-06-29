from copy import deepcopy
from settings import *
from abc import ABC, abstractmethod
from shapes.parametric import Parametric
from shapes.point import Point
from shapes.line import EuclideanLine as Line
from settings import SCALE

class Tessellation(ABC):
    def __init__(self, width=DEFAULT_THICKNESS, stroke=DEFAULT_STROKE, **kwargs):
        self.lines = []
        self.width, self.stroke = width, stroke

    def draw(self, svg):
        for line in self.lines:
            line.draw(svg)

    def set_stroke(self, stroke):
        self.stroke = stroke
    
    def conformal(self, f):
        new_tess = deepcopy(self)
        new_tess.lines = [line.conformal(f) for line in self.lines]
        return new_tess


class MiuraOri(Tessellation):
    def __init__(self, layers, depth, parametric=False, parametric_verts=False, parametric_diags=False, squash_factor=1, top=True, h_width=SCALE[0]*2, v_height=SCALE[1]*2, **kwargs):
        super().__init__(**kwargs)
        start = kwargs.get("start", Point(-h_width/2, -v_height/2))
        h = h_width/layers
        k = v_height/depth

        self.lines = []
        for i in range(layers + 1):
            vp1, vp2 = Point(start.x + i*h, start.y), Point(start.x + i*h, start.y + v_height)
            if parametric or parametric_verts: self.lines.append(Parametric(vp1, vp2, stroke=self.stroke, width=self.width))
            else: self.lines.append(Line(vp1, vp2, stroke=self.stroke, width=self.width))
            j = 0
            while j < depth and i < layers:
                dx1, dx2 = Point(start.x + i*h, start.y + j*k + (-1/2 + (i % 2))*k*squash_factor+k/2), Point(start.x + (i + 1)*h, start.y + j*k + (-1/2 + ((i + 1) % 2))*k*squash_factor+k/2)
                if parametric or parametric_diags: self.lines.append(Parametric(dx1, dx2, stroke=self.stroke, width=self.width))
                else: self.lines.append(Line(dx1, dx2, stroke=self.stroke, width=self.width))
                j += 1
        if top:
            for i in range(2): self.lines.append(Line(Point(start.x, start.y + i*v_height), Point(start.x + h_width, start.y + i*v_height), stroke=self.stroke, width=self.width))


    
class Waterbomb(Tessellation):
    def __init__(self, layers, depth, parametric=False, parametric_verts=False, parametric_diags=False, squash_factor=1, top=True, h_width=SCALE[0]*2, v_height=SCALE[1]*2, **kwargs):
        super().__init__(**kwargs)
        start = kwargs.get("start", Point(-h_width/2, -v_height/2))
        h = h_width/layers
        k = v_height/depth

        self.lines = []
        for i in range(layers + 1):
            vp1, vp2 = Point(start.x + i*h, start.y), Point(start.x + i*h, start.y + v_height)
            if parametric or parametric_verts: self.lines.append(Parametric(vp1, vp2, stroke=self.stroke, width=self.width))
            else: self.lines.append(Line(vp1, vp2, stroke=self.stroke, width=self.width))
            j = 0
            while j < depth and i < layers:
                dx1, dx2 = Point(start.x + i*h, start.y + j*k + (-1/2 + (i % 2))*k*squash_factor+k/2), Point(start.x + (i + 1)*h, start.y + j*k + (-1/2 + ((i + 1) % 2))*k*squash_factor+k/2)
                if parametric or parametric_diags: self.lines.append(Parametric(dx1, dx2, stroke=self.stroke, width=self.width))
                else: self.lines.append(Line(dx1, dx2, stroke=self.stroke, width=self.width))
                j += 1
        if top:
            for i in range(2): self.lines.append(Line(Point(start.x, start.y + i*v_height), Point(start.x + h_width, start.y + i*v_height), stroke=self.stroke, width=self.width))
