from copy import deepcopy
from svgpathtools import svg2paths
from shapes import EuclideanLine as Line
from shapes import Point
from settings import SCALE
import re

class ImportedSVG():
    def __init__(self, path, pivot=Point(0,0), rel_scale="auto"):
        self.pivot, self.rel_scale = pivot, rel_scale
        if self.rel_scale != "auto":
            if not re.match(r"\(\d+(.\d+)?[u%],\d+(.\d+)?[u%]\)$", self.rel_scale): raise ValueError("Incorrect relative scale for imported SVG")
            self.rel_scale = self.rel_scale[1:-1].split(',')
            for i in range(len(self.rel_scale)):
                if self.rel_scale[i][-1] == "%": self.rel_scale[i] = float(self.rel_scale[i][:-1])/100
                else: self.rel_scale[i] = float(self.rel_scale[i][:-1])
            

        _, attributes = svg2paths(path)
        self.lines = []
        for v in attributes:
            point_strings = v['d'].split(' ')
            valid_points = []
            state = point_strings[0]
            i = 1
            while i < len(point_strings):
                if point_strings[i].isalpha():
                    state = point_strings[i]
                    if state.lower() == "z": valid_points.append(valid_points[0])
                    i += 1
                    continue

                if len(valid_points) > 0:
                    prevX, prevY = valid_points[-1].split(",")
                if state == "V": newpoint = f"{prevX},{point_strings[i]}"
                elif state == "v": newpoint = f"{prevX},{float(point_strings[i]) + float(prevY)}"
                elif state == "H": newpoint = f"{point_strings[i]},{prevY}"
                elif state == "h": newpoint = f"{float(point_strings[i]) + float(prevX)},{prevY}"
                elif state == "l":
                    rel_coords = point_strings[i].split(',')
                    newpoint = f"{float(rel_coords[0]) + float(prevX)},{float(rel_coords[1]) + float(prevY)}"
                elif state == "m" and i > 1: 
                    rel_coords = point_strings[i].split(',')
                    newpoint = f"{float(rel_coords[0]) + float(prevX)},{float(rel_coords[1]) + float(prevY)}"
                elif state == "M" or state == "L" or i == 1:
                    newpoint = point_strings[i]
                valid_points.append(newpoint)
                i += 1
            
            parts = [string.split(',') for string in valid_points if not any(char.isalpha() for char in string) and ',' in string]
            for i in range(len(parts) - 1):
                self.lines.append(Line(Point(float(parts[i][0]), float(parts[i][1])), Point(float(parts[i+1][0]), float(parts[i+1][1]))))

        self.center, self.scale = self.get_center()

    def draw(self, svg):
        for line in self.lines:
            line.draw(svg)

    def get_center(self):
        points = []
        for line in self.lines:
            points.append(line.start)
            points.append(line.end)
        
        min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
        
        for point in points:
            min_x, max_x, min_y, max_y = min(min_x, point.x), max(max_x, point.x), min(min_y, point.y), max(max_y, point.y)

        max_dist = max(max_x - min_x, max_y - min_y)
        
        if self.rel_scale == "auto": return Point((min_x + max_x)/2, (min_y + max_y)/2), (SCALE[0]/max_dist*2, SCALE[1]/max_dist*2)
        return Point((min_x + max_x)/2, (min_y + max_y)/2), (2*self.rel_scale[0]/abs(min_x - max_x), 2*self.rel_scale[1]/abs(min_y - max_y))
    

    def set_center(self, point: Point):
        self.center = point

    def generate_paths(self):
        paths = []
        for line in self.lines:
            new_line = deepcopy(line)
            new_line.translate(self.center*Point(-1))
            new_line.scale((self.scale[0], -self.scale[1]))
            new_line.translate(self.pivot)
            paths.append(new_line)
        return paths
