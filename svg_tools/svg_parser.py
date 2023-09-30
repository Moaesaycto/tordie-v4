from copy import deepcopy
from shapes import EuclideanLine as Line
from shapes import Point, Parametric
from settings import SCALE
import re
# requires svg.path, install it like this: pip3 install svg.path
from svg.path import parse_path
from xml.dom import minidom

from svg.path.path import Line as SVGLine
from svg.path.path import CubicBezier, Move, Close
from math import floor, pi

from math import exp as e # Built in maths class
from math import floor, pi
from helpers import  exp

class ImportedSVG():
    def __init__(self, path, pivot=Point(0,0), rel_scale="auto", euclidean=True):
        self.mapping_eligible = True
        self.euclidean = euclidean
        self.path = path
        self.pivot, self.rel_scale = pivot, rel_scale
        if self.rel_scale != "auto":
            if not re.match(r"\(\d+(.\d+)?[u%],\d+(.\d+)?[u%]\)$", self.rel_scale): raise ValueError("Incorrect relative scale for imported SVG")
            self.rel_scale = self.rel_scale[1:-1].split(',')
            for i in range(len(self.rel_scale)):
                if self.rel_scale[i][-1] == "%": self.rel_scale[i] = float(self.rel_scale[i][:-1])/100
                else: self.rel_scale[i] = float(self.rel_scale[i][:-1])

        self.generate_lines()
        self.center, self.scale = self.get_center()
        self.generate_lines(center=self.center, scale=self.scale, pivot=pivot)

    def generate_lines(self, center=Point(0,0), scale=(1,1), pivot=Point(0,0)):
        doc = minidom.parse(self.path)
        path_strings = [path.getAttribute('d') for path in doc.getElementsByTagName('path')]
        doc.unlink()

        if len(path_strings) > 1 or path_strings[-1][-1] not in "zZ": self.mapping_eligible = False

        abs_paths = []
        for path_string in path_strings:
            path = parse_path(path_string)
            for e in path: abs_paths.append(e)

        self.lines = []
        for i in range(len(abs_paths)):
            path = abs_paths[i]
            if isinstance(path, CubicBezier):
                startx, starty = deepcopy(path.start.real), path.start.imag, 
                dx1, dy1 = path.control1.real, path.control1.imag
                dx2, dy2 = path.control2.real, path.control2.imag
                dx, dy = path.end.real, path.end.imag
                self.lines.append(Parametric(
                    lambda t, startx=startx, dx1=dx1, dx2=dx2, dx=dx: ((1-t)**3*startx + 3*t*(1-t)**2*dx1 + 3*t**2*(1-t)*dx2 + t**3*dx - center.x)*scale[0] + pivot.x, 
                    lambda t, starty=starty, dy1=dy1, dy2=dy2, dy=dy: (-(1-t)**3*starty - 3*t*(1-t)**2*dy1 - 3*t**2*(1-t)*dy2 - t**3*dy - center.y)*scale[1] + pivot.y,
                      0, 1))

            elif isinstance(path, SVGLine) or (isinstance(path, Close) and isinstance(abs_paths[i-1], SVGLine)):
                if self.euclidean: self.lines.append(Line(
                    (Point(path.start.real, -path.start.imag) - center)*scale[0] + pivot.x, 
                    (Point(path.end.real, -path.end.imag) - center)*scale[1] + pivot.y
                ))
                else: self.lines.append(Parametric(
                    (Point(path.start.real, -path.start.imag) - center)*scale[0] + pivot.x, 
                    (Point(path.end.real, -path.end.imag) - center)*scale[1] + pivot.y
                ))
            elif isinstance(path, Move) or isinstance(path, Close): pass

    def draw(self, svg):
        for line in self.lines:
            line.draw(svg)

    def get_center(self):
        points = []
        for line in self.lines:
            if isinstance(line, Parametric):
                for point in line.points.values(): points.append(point)
            else:
                points.append(line.start)
                points.append(line.end)

        min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
        
        for point in points:
            min_x, max_x, min_y, max_y = min(min_x, point.x), max(max_x, point.x), min(min_y, point.y), max(max_y, point.y)

        max_dist = max(max_x - min_x, max_y - min_y)
        
        if self.rel_scale == "auto":
            if max_dist == 0: return Point((min_x + max_x)/2, (min_y + max_y)/2), (1, 1)
            return Point((min_x + max_x)/2, (min_y + max_y)/2), (SCALE[0]/max_dist*2, SCALE[1]/max_dist*2)
        return Point((min_x + max_x)/2, (min_y + max_y)/2), (2*self.rel_scale[0]/abs(min_x - max_x), 2*self.rel_scale[1]/abs(min_y - max_y))

    def generate_circular_mapping(self):
        fx = lambda t, lines=self.lines: lines[floor(t)-1].fx(t - floor(t))
        fy = lambda t, lines=self.lines: lines[floor(t)-1].fy(t - floor(t))
        us_contour = Parametric(fx, fy, 0, len(self.lines)).reparameterize_unit_speed()
        cm = lambda p: p.x/e(p.x)*exp(p)
        return lambda t: us_contour.f(cm(t).theta360*us_contour.tn/(2*pi) % us_contour.tn)*cm(t).r
        
    def conformal(self, f):
        paths = self.generate_paths()
        return [path.conformal(f) for path in paths]
        