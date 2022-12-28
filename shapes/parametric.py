from svgwrite.drawing import Drawing
from svgwrite import shapes
from copy import deepcopy

from maths.helpers import *
from shapes.point import Point
from shapes.line import Line
from shapes.circle import Circle
from options import *

class Parametric(Line):
    def __init__(self, fx, fy, t0, tn, **kwargs):
        super().__init__(Point(fx(t0), fy(t0)), Point(fx(tn), fy(tn)), **kwargs)
        h = kwargs.get("step", DEFAULT_STEP)
        self.observer = kwargs.get("observer", Point(0, 0))
        self.fx, self.fy = fx, fy
        self.t0, self.tn, self.h = t0, tn, h

        self.points = {}
        i = t0
        while i <= tn:
            self.points[i] = Point(fx(i), fy(i))
            i += (tn - t0)/h
        if i != tn: self.points[i] = Point(fx(tn), fy(tn))


    def draw(self, svg: Drawing):
        adjusted_points = [point.adjust(svg.width, svg.height, SCALE) for point in self.points.values()]
        result = svg.drawing.add(shapes.Polyline(adjusted_points, stroke_width=self.thickness,stroke=self.stroke))
        result.fill('white', opacity=0)


    def reflect(self, point: Point):
        overPoint = self.getReflectPoint(point)
        for i in self.points.keys():
            if self.points[i] == overPoint:
                tk = i
                break
        else: return None

        a1, a2 = diff(self.fx, tk), diff(self.fx, tk + DIFF_STEP) 
        b1, b2 = diff(self.fy, tk), diff(self.fy, tk + DIFF_STEP)
        c1, c2 = self.fx(tk)*a1 + self.fy(tk)*b1, self.fx(tk + DIFF_STEP)*a2 + self.fy(tk + DIFF_STEP)*b2

        c = Point((-b1*c2 + c1*b2)/(a1*b2 - b1*a2), (a1*c2 - c1*a2)/(a1*b2 - b1*a2))
        return Circle(c, dist(c, overPoint)).reflect(point)


    def getReflectPoint(self, point: Point):
        current = None
        for p in self.points.values():
            if p.x == point.x or self.observer.x == point.x:
                if (current is None or dist(p, self.observer) < dist(current, self.observer)) and (p.x == point.x and self.observer.x == point.x):
                    current = p
                continue
            elif abs((p.y - point.y)/(p.x - point.x) - (self.observer.y - point.y)/(self.observer.x - point.x)) < PARAM_REFLECT_M_DIFF:
                if current is None or dist(p, self.observer) < dist(current, self.observer):
                    current = p
        return current
    

    def paraReflect(self, parametric):
        reflection = deepcopy(parametric)
        reflection.points = {}
        for t in parametric.points.keys():
            tempPoint = self.reflect(parametric.points[t])
            if tempPoint is not None: reflection.points[t] = tempPoint
        return reflection

    def conformal(self, f):
        result = deepcopy(self)
        for t in self.points.keys():
         result.points[t] = f(self.points[t])
        return result
