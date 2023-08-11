from abc import ABC, abstractmethod
from typing import Union
from math import sin, cos, sqrt, atan2, acos, radians, pi
from svgwrite.drawing import Drawing
import numpy as np
from utils import color
from numpy import sqrt
from svgwrite import shapes
from copy import deepcopy
from settings import *

class Shape(ABC):
    def __init__(self, width=DEFAULT_THICKNESS, stroke=DEFAULT_STROKE):
        self.width = width
        self.stroke = stroke
    
    @abstractmethod
    def draw(svg): pass

    @abstractmethod
    def translate(vector): pass

    @abstractmethod
    def rotate(angle, *args): pass

    @abstractmethod
    def scale(scale, *args): pass

    @abstractmethod
    def intersect(self, other): pass
    
    def set_stroke(self, stroke: str): self.stroke = stroke

    def set_width(self, width: Union[int, float]): self.width = width


class Point(Shape):
    def __init__(self, coord1, *args, polar=False, radius=DEFAULT_RADIUS, **kwargs):
        """
        Assumes Cartesian unless 'polar=True', where coord1 is r and coord2 is theta
        """
        super().__init__(**kwargs)
        self.radius = radius
        coord2 = args[0] if args else 0
        if polar:
            self.x, self.y = coord1*cos(coord2), coord1*sin(coord2)
            self.r, self.theta = coord1, coord2
        else:
            self.x, self.y = coord1, coord2
            self.r, self.theta = sqrt(coord1**2 + coord2**2), atan2(self.y, self.x)

    def __eq__(self, other):
        return isinstance(other, Point) and self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __add__(self, other):
        if isinstance(other, Point): return Point(self.x + other.x, self.y + other.y)
        if isinstance(other, (int, float)): return Point(self.x + other, self.y + other)
    
    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, Point): return Point(self.x - other.x, self.y - other.y)
        if isinstance(other, (int, float)): return Point(self.x - other, self.y - other)

    def __pow__(self, other):
        arg, mod = self.theta*other, self.r**other
        return Point(mod*np.cos(arg), mod*np.sin(arg))
    
    def __mul__(self, other):
        if isinstance(other, Point): return Point(self.x*other.x - self.y*other.y, self.x*other.y + self.y*other.x)
        if isinstance(other, (int, float)): return Point(self.x * other, self.y * other) # Scalar multiplication

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, Point): return Point(self.x*other.x + self.y*other.y, -self.x*other.y + self.y*other.x) / (other.x**2 + other.y**2)
        if isinstance(other, Union[int, float]): return Point(self.x / other, self.y / other)

    def __rtruediv__(self, other):
        return (self / other)**(-1) if other != 0 else 0

    def __abs__(self):
        return self.r
    
    def __repr__(self):
        return f"Point({self.x}, {self.y})"
     
    def set_cartesian(self, x, y):
        self.x, self.y = x, y
        self.r, self.theta = sqrt(x**2 + y**2), atan2(y, x)
    
    def set_polar(self, r, theta):
        self.x, self.y = r*cos(theta), r*sin(theta)
        self.r, self.theta = r, theta

    def adjust(self, width, height, scale):
        return (
            self.x / 2* float(width / scale[0]) + (width + BOUNDS_FIX) / 2,
            -self.y / 2 * float(height / SCALE[1]) + (height + BOUNDS_FIX)/ 2
        )

    def draw(self, svg: Drawing):
        disk = svg.drawing.add(svg.drawing.circle(self.adjust(svg.width, svg.height, SCALE), self.radius))
        disk.fill(self.stroke).stroke(self.stroke, width=0)

    def reflect(self, point):
        new_point = deepcopy(point)
        new_point.set_cartesian(2*self.x - point.x, 2*self.y - point.y)
        return new_point

    def translate(self, vector):
        self.set_cartesian(self.x + vector.x, self.y + vector.y)

    def rotate(self, angle, *args):
        if angle == 0: return
        if not isinstance(args[0], Point): raise TypeError("Center for rotation must be a Point")
        center = args[0] if args else Point(0, 0)
        self.x, self.y = self.x - center.x, self.y - center.y

        prev_x, prev_y = self.x, self.y
        if angle != 0:
            self.x = prev_x*cos(radians(angle)) - prev_y*sin(radians(angle))
            self.y = prev_y*cos(radians(angle)) + prev_x*sin(radians(angle))
    
        self.set_cartesian(self.x + center.x, self.y + center.y)

    def scale(self, scale, *args):
        if scale == (1, 1) or scale == 1: return
        center = args[0] if args else Point(0, 0)
        if not isinstance(scale, (tuple, list)): scale = (scale, scale)
        self.set_cartesian(scale[0]*(self.x - center.x) + center.x, scale[1]*(self.y - center.y) + center.y)

    def midpoint(self, point): return Point((self.x + point.x)/2, (self.y + point.y)/2)

    def intersect(self, other): pass # TODO


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
    
    def intersect(self, other):
        # Intersection with other lines
        if isinstance(other, EuclideanLine): return self.intersect_euclidean_line(other)
        if isinstance(other, PoincareLine): pass
        if isinstance(other, Parametric): other.intersect(self)

        if isinstance(other, Circle): return self.intersect_circle(other)
        if isinstance(other, Polygon): return other.intersect(self)
    

class EuclideanLine(Line):
    def __init__(self, point1, point2, true_line=False, **kwargs):
        if true_line:
            m = (point2.y - point1.y)/(point2.x - point1.x) if (point2.x - point1.x) != 0 else float('inf')
            b = point1.y - m*point1.x
            point1 = Point(-(SCALE[0]+BOUNDS_FIX*2), -(SCALE[0]+BOUNDS_FIX*2)*m + b)
            point2 = Point(SCALE[0]+BOUNDS_FIX*2, (SCALE[0]+BOUNDS_FIX*2)*m + b)
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

    def intersect_euclidean_line(self, other):
        x11, y11 = self.start.x, self.start.y
        x12, y12 = self.end.x, self.end.y
        x21, y21 = other.start.x, other.start.y
        x22, y22 = other.end.x, other.end.y

        slope1 = (y12 - y11) / (x12 - x11) if (x12 - x11) != 0 else float('inf')
        slope2 = (y22 - y21) / (x22 - x21) if (x22 - x21) != 0 else float('inf')
        if slope1 == slope2: return []

        intercept1 = y11 - slope1 * x11
        intercept2 = y21 - slope2 * x21

        if slope1 == float('inf'): x_intersect = x11
        elif slope2 == float('inf'): x_intersect = x21
        else: x_intersect = (intercept2 - intercept1) / (slope1 - slope2)

        if slope1 == float('inf'): y_intersect = slope2 * x_intersect + intercept2
        else: y_intersect = slope1 * x_intersect + intercept1

        if (
            min(x11, x12) <= x_intersect <= max(x11, x12) and
            min(x21, x22) <= x_intersect <= max(x21, x22) and
            min(y11, y12) <= y_intersect <= max(y11, y12) and
            min(y21, y22) <= y_intersect <= max(y21, y22)
        ): return [Point(x_intersect, y_intersect)]
        else: return []

    def intersect_circle(self, other):
        x1, y1 = self.start.x, self.start.y
        x2, y2 = self.end.x, self.end.y
        xc, yc = other.center.x, other.center.y
        r = other.radius

        dx = x2 - x1
        dy = y2 - y1
        a = dx**2 + dy**2
        b = 2 * (dx * (x1 - xc) + dy * (y1 - yc))
        c = (x1 - xc)**2 + (y1 - yc)**2 - r**2
        discriminant = b**2 - 4 * a * c

        if discriminant < 0: return []

        elif discriminant == 0:
            t = -b / (2 * a)
            if 0 <= t <= 1:
                x_intersect = x1 + t * dx
                y_intersect = y1 + t * dy
                return [Point(x_intersect, y_intersect)]

        else:
            t1 = (-b + sqrt(discriminant)) / (2 * a)
            t2 = (-b - sqrt(discriminant)) / (2 * a)

            intersection_points = []
            if 0 <= t1 <= 1:
                x_intersect1 = x1 + t1 * dx
                y_intersect1 = y1 + t1 * dy
                intersection_points.append((x_intersect1, y_intersect1))

            if 0 <= t2 <= 1:
                x_intersect2 = x1 + t2 * dx
                y_intersect2 = y1 + t2 * dy
                intersection_points.append((x_intersect2, y_intersect2))

            if intersection_points:
                return [Point(i[0], i[1]) for i in intersection_points]
        return []


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

    def translate(self, vector):
        pass

    def rotate(self, angle, *args):
        pass

    def scale(self, scale, *args):
        pass

    def intersect(self, other):
        if isinstance(other, Circle): return self.intersect_circle(other)
        elif isinstance(other, Line): return other.intersect(self)
        elif isinstance(other, Circle): return other.intersect(self)
    
    def intersect_circle(self, circle):
        xc1, yc1 = self.center.x, self.center.y
        xc2, yc2 = circle.center.x, circle.center.y
        r1, r2 = self.radius, circle.radius

        dx = xc2 - xc1
        dy = yc2 - yc1
        distance = sqrt(dx**2 + dy**2)

        if distance > r1 + r2 or distance < abs(r1 - r2): return []

        alpha = atan2(dy, dx)
        beta = acos((r1**2 + distance**2 - r2**2) / (2 * r1 * distance))

        intersection_angle1 = alpha + beta
        intersection_angle2 = alpha - beta
        x_intersect1 = xc1 + r1 * cos(intersection_angle1)
        y_intersect1 = yc1 + r1 * sin(intersection_angle1)
        x_intersect2 = xc1 + r1 * cos(intersection_angle2)
        y_intersect2 = yc1 + r1 * sin(intersection_angle2)

        if intersection_angle1 == intersection_angle2: return [Point(x_intersect1, y_intersect1)]
        else: return [Point(x_intersect1, y_intersect1), Point(x_intersect2, y_intersect2)]

    def conformal(self, f): return Parametric(self).conformal(f)


class Parametric(Line):
    def __init__(self, *args, observer=Point(0,0), step=DEFAULT_STEP, **kwargs):
        """
        Args 1: fx, fy, t0, tn
        Args 2: p1, p2
        """
        h = step
        self.observer = observer
        if len(args) == 4:
            fx, fy, t0, tn = args[0], args[1], args[2], args[3]
            assert callable(fx) and callable(fy), "fx and fy must be callable"
            assert isinstance(t0, (int, float)) and isinstance(tn, (int, float)), "t0 and tn must be numbers"
            assert t0 < tn, "t0 must be less than tn"
            super().__init__(Point(fx(t0), fy(t0)), Point(fx(tn), fy(tn)), **kwargs)
            self.fx, self.fy = fx, fy
            self.t0, self.tn, self.h = t0, tn, h
        
        elif len(args) == 2:
            p1, p2 = args[0], args[1]
            assert isinstance(p1, Point) and isinstance(p2, Point), "p1 and p2 must be points"
            super().__init__(p1, p2, **kwargs)
            fx, fy = lambda t: p1.x + t*(p2.x - p1.x), lambda t: p1.y + t*(p2.y - p1.y)
            t0, tn = 0, 1
            self.points = {0: p1, 1: p2}
        
        elif len(args) == 1:
            if isinstance(args[0], Line):
                line = args[0]
                assert isinstance(line, Line), "Argument must be a line"
                super().__init__(line.start, line.end, **kwargs)
                fx, fy = lambda t: line.start.x + t*(line.end.x - line.start.x), lambda t: line.start.y + t*(line.end.y - line.start.y)
                t0, tn = 0, 1
                self.points = {0: line.start, 1: line.end}
            
            if isinstance(args[0], Circle):
                circle = args[0]
                assert isinstance(circle, Circle), "Argument must be a circle"
                return self.__init__(lambda t: circle.radius*cos(t) + circle.center.x, lambda t: circle.radius*sin(t) + circle.center.y, 0, 2*pi, observer=observer, step=step, **kwargs)

        else: raise TypeError("Invalid arguments")
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
        over_point = self.get_reflect_point(point)
        for i in self.points.keys():
            if self.points[i] == over_point:
                tk = i
                break
        else: return None

        a1, a2 = diff(self.fx, tk), diff(self.fx, tk + DIFF_STEP) 
        b1, b2 = diff(self.fy, tk), diff(self.fy, tk + DIFF_STEP)
        c1, c2 = self.fx(tk)*a1 + self.fy(tk)*b1, self.fx(tk + DIFF_STEP)*a2 + self.fy(tk + DIFF_STEP)*b2

        c = Point((-b1*c2 + c1*b2)/(a1*b2 - b1*a2), (a1*c2 - c1*a2)/(a1*b2 - b1*a2))
        return Circle(c, dist(c, over_point)).reflect(point)


    def get_reflect_point(self, point: Point):
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
    

    def para_reflect(self, parametric):
        reflection = deepcopy(parametric)
        reflection.points = {}
        for t in parametric.points.keys():
            tempPoint = self.reflect(parametric.points[t])
            if tempPoint is not None: reflection.points[t] = tempPoint
        return reflection


    def conformal(self, f):
        result = deepcopy(self)
        for t in self.points.keys(): result.points[t] = f(self.points[t])
        return result

    def intersect(self, other):
        if isinstance(other, EuclideanLine): return self.intersect_euclidean_line(other)
        elif isinstance(other, Parametric): return self.intersect_parametric_parametric(other)
        elif isinstance(other, Circle): return self.intersect_circle(other)
        elif isinstance(other, Polygon): return other.intersect(self)
    
    def intersect_euclidean_line(self, line):
        intersection_points = []
        points = [point for point in self.points.values()]
        for i in range(len(points) - 1):
            curr_start, curr_end = points[i], points[i + 1]
            curr_line = EuclideanLine(curr_start, curr_end)
            intersection_points += self.euclidean_line_euclidean_line(line, curr_line)
        return intersection_points
    
    def intersect_parametric(self, parametric):
        intersection_points = []
        points1 = [point for point in self.points.values()]
        points2 = [point for point in parametric.points.values()]
        for i in range(len(points1) - 1):
            curr_start1, curr_end1 = points1[i], points1[i + 1]
            curr_line1 = EuclideanLine(curr_start1, curr_end1)
            for j in range(len(points2) - 1):
                curr_start2, curr_end2 = points2[j], points2[j + 1]
                curr_line2 = EuclideanLine(curr_start2, curr_end2)
                intersection_points += curr_line1.euclidean_line_parametric(curr_line2)

        return intersection_points

    def intersect_circle(self, circle):
        intersection_points = []
        points = [point for point in self.points.values()]
        for i in range(len(points) - 1):
            curr_start, curr_end = points[i], points[i + 1]
            curr_line = EuclideanLine(curr_start, curr_end)
            intersection_points += curr_line.intersect_circle(circle)

        return intersection_points


class Polygon(Shape):
    def __init__(self, points, parametric=False, m=0, center=None, m_center="relative", show_points=False, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs
        if len(points) < 3: raise ValueError("Polygons must have at least 3 points")
        self.raw_points, self.n = points, len(points)
        self._centroid = sum(points)/len(points)

        self.winding_number = 0
        for i in range(len(points)): self.winding_number += (points[(i + 1) % self.n].x - points[i].x)*(points[(i + 1) % self.n].y + points[i].y)

        self.parametric = parametric
        self.m = [m] if isinstance(m, (int, float)) else m

        self.min_x, self.min_y, self.max_x, self.max_y = float("inf"), float("inf"), -float("inf"), -float("inf")
        for point in self.raw_points:
            self.min_x, self.min_y = min(self.min_x, point.x), min(self.min_y, point.y)
            self.max_x, self.max_y = max(self.max_x, point.x), max(self.max_y, point.y)

        self.horizontal_width, self.vertical_height = self.max_x - self.min_x, self.max_y - self.min_y
        
        if any(x < -1 for x in self.m): raise ValueError("m must always be greater than or equal to -1")
        self.points = self.get_points(m_center=m_center)
        self.points.append(self.points[0])
        self.sides = self.get_side(self.parametric, **kwargs)
        self.show_points = show_points
        if center is not None: self._centroid.translate(center - self._centroid)

    def get_points(self, m_center="auto"):
        if all(x == 0 for x in self.m): return self.raw_points
        m_points = []

        match m_center:
            case "auto":
                for i in range(self.n):
                    m_points.append(self.raw_points[i])
                    b_point = self.raw_points[i].midpoint(self.raw_points[(i + 1) % self.n])
                    b_point.scale(self.m[i % len(self.m)] + 1, self._centroid)
                    m_points.append(b_point)
            case "relative":
                for i in range(self.n):
                    p1, p2 = self.raw_points[i], self.raw_points[(i + 1) % self.n]
                    m_points.append(p1)
                    diff = p2 - p1
                    dir_vec = Point(-diff.y, diff.x)
                    pm = p1.midpoint(p2)
                    m_points.append(self.m[i % len(self.m)]*self.winding_number*dir_vec*0.5/abs(self.winding_number) + pm)
            
        m_points.append(self.raw_points[0])
        return m_points

    def get_side(self, parametric, **kwargs):
        if parametric: return [Parametric(self.points[i], self.points[i + 1], **kwargs) for i in range(len(self.points) - 1)]
        else: return [EuclideanLine(self.points[i], self.points[i + 1], **kwargs) for i in range(len(self.points) - 1)]

    def draw(self, svg):
        for side in self.sides: side.draw(svg)
        if self.show_points:
            for point in self.raw_points: point.draw(svg)
    
    def translate(self, vector):
        self.points, self.sides = [point.translate(vector) for point in self.points], [side.translate(vector) for side in self.sides]

    def rotate(self, angle, *args): 
        self.points, self.sides = [point.rotate(angle, *args) for point in self.points], [side.rotate(angle, *args) for side in self.sides]

    def scale(self, scale, *args):
        self.points, self.sides = [point.scale(scale, *args) for point in self.points], [side.scale(scale, *args) for side in self.sides]
    
    def conformal(self, f):
        self.points, self.sides = [f(point) for point in self.points], [side.conformal(f) for side in self.sides]
    
    def pnpoly(self, point):
        x, y = point.x, point.y
        inside = False

        for i in range(self.n + 1):
            j = i - 1
            if ((self.points[i % self.n].y > y) != (self.points[j].y > y)) and (x < (self.points[j].x - self.points[i % self.n].x) * (y - self.points[i % self.n].y) / (self.points[j].y - self.points[i % self.n].y) + self.points[i % self.n].x):
                inside = not inside
        return inside
    
    def intersect(self, other):
        if isinstance(other, Polygon): return self.intersect_polygon(other)
        return [item for sublist in [side.intersect(other) for side in self.sides] for item in sublist]
    
    def intersect_polygon(self, other):
        intersection_points = []
        for side1 in self.sides:
            for side2 in other.sides:
                intersection_points += side1.intersect(side2)
        return intersection_points


class RegularPolygon(Polygon):
    def __init__(self, sides, center=Point(0, 0), radius=1, apothem=None, rotation=0, **kwargs):
        if sides < 3: raise ValueError("A polygon must have at least 3 sides")

        if apothem is None: apothem = radius * cos(pi / sides)
        else: radius = apothem / cos(pi / sides)
        if radius <= 0: raise ValueError("Radius/apothem must be positive")
        self.center, self._centroid = center, center
        self.radius, self.apothem = radius, apothem
        self.rotation = rotation

        main_points = [Point(self.radius, 2 * pi * i / sides + self.rotation, polar=True) + center for i in range(sides + 1)]
        super().__init__(main_points, center=center, **kwargs)

# CHECK THESE!!!!!
class Triangle(Polygon):
    def __init__(self, point1, point2, point3, **kwargs):
        super().__init__([point1, point2, point3], **kwargs)
        self.p1, self.p2, self.p3 = point1, point2, point3
    
    def circumcircle(self, **kwargs):
        circumcenter = self.circumcenter()
        return Circle(circumcenter, dist(circumcenter, self.points[0]), **kwargs)
    
    def incircle(self, **kwargs):
        A, B, C = self.p1, self.p2, self.p3
        a, b, c = dist(B, C), dist(A, C), dist(A, B)
        s = (a + b + c)/2
        r = sqrt((s - a)*(s - b)*(s - c)/s)
        return Circle(self.incenter(), r, **kwargs)
    
    def orthocenter(self, **kwargs):
        A, B, C = self.p1, self.p2, self.p3
        x_num = (A.y**2*(C.y - B.y) + B.x*C.x*(C.y - B.y) + B.y**2*(A.y - C.y) + A.x*C.x*(A.y - C.y) + C.y**2*(B.y - A.y) + A.x*B.x*(B.y - A.y))
        y_num = (A.x**2*(B.x - C.x) + B.y*C.y*(B.x - C.x) + B.x**2*(C.x - A.x) + A.y*C.y*(C.x - A.x) + C.x**2*(A.x - B.x) + A.y*B.y*(A.x - B.x))
        denom = A.y*(C.x - B.x) + B.y*(A.x - C.x) + C.y*(B.x - A.x)
        return Point(x_num/denom, y_num/denom, **kwargs)
    
    def centroid(self, **kwargs):
        return Point((self.p1.x + self.p2.x + self.p3.x)/3, (self.p1.y + self.p2.y + self.p3.y)/3, **kwargs)
    
    def incenter(self, **kwargs):
        A, B, C = self.p1, self.p2, self.p3
        a, b, c = dist(B, C), dist(A, C), dist(A, B)
        return Point((a*A.x + b*B.x + c*C.x)/(a + b + c), (a*A.y + b*B.y + c*C.y)/(a + b + c), **kwargs)
    
    def circumcenter(self, **kwargs):
        A, B, C = self.p1, self.p2, self.p3
        x_num = ((A.x**2 + A.y**2)*(B.y - C.y) + (B.x**2 + B.y**2)*(C.y - A.y) + (C.x**2 + C.y**2)*(A.y - B.y))
        y_num = ((A.x**2 + A.y**2)*(C.x - B.x) + (B.x**2 + B.y**2)*(A.x - C.x) + (C.x**2 + C.y**2)*(B.x - A.x))
        denom = 2*(A.x*(B.y - C.y) + B.x*(C.y - A.y) + C.x*(A.y - B.y))
        return Point(x_num/denom, y_num/denom, **kwargs)
    
    def euler_line(self, **kwargs):
        return EuclideanLine(self.centroid(), self.orthocenter(), true_line=True, **kwargs)
    
    def nine_point_circle(self):
        return Circle(self.euler_line().midpoint(self.circumcenter()), self.circumcircle().radius/2)
    
    def medial_triangle(self):
        return Triangle(*[side.midpoint() for side in self.sides])
    
    def orthic_triangle(self):
        return Triangle(*[self.points[i].reflect(self.sides[i]) for i in range(3)])
    
    def pedal_triangle(self, point):
        return Triangle(*[point.reflect(side) for side in self.sides])
    
    # https://www.wikiwand.com/en/Incircle_and_excircles
    def excenters(self):
        A, B, C = self.p1, self.p2, self.p3
        a, b, c = dist(B, C), dist(A, C), dist(A, B)
        return [Point(a*A.x + b*B.x + c*C.x, a*A.y + b*B.y + c*C.y)/(a + b + c), Point(-a*A.x + b*B.x + c*C.x, -a*A.y + b*B.y + c*C.y)/(a + b + c), Point(a*A.x - b*B.x + c*C.x, a*A.y - b*B.y + c*C.y)/(a + b + c)]
    
    def exradii(self):
        return [Circle(self.points[i], dist(self.points[i], self.excenters()[i])) for i in range(3)]
    
    def excircles(self):
        return [Circle(self.points[i], dist(self.points[i], self.excenters()[i])) for i in range(3)]


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
            else: self.lines.append(EuclideanLine(vp1, vp2, stroke=self.stroke, width=self.width))
            j = 0
            while j < depth and i < layers:
                dx1, dx2 = Point(start.x + i*h, start.y + j*k + (-1/2 + (i % 2))*k*squash_factor+k/2), Point(start.x + (i + 1)*h, start.y + j*k + (-1/2 + ((i + 1) % 2))*k*squash_factor+k/2)
                if parametric or parametric_diags: self.lines.append(Parametric(dx1, dx2, stroke=self.stroke, width=self.width))
                else: self.lines.append(EuclideanLine(dx1, dx2, stroke=self.stroke, width=self.width))
                j += 1
        if top:
            for i in range(2): self.lines.append(EuclideanLine(Point(start.x, start.y + i*v_height), Point(start.x + h_width, start.y + i*v_height), stroke=self.stroke, width=self.width))

    
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


"""
HELPERS FOR INITIALISATION
"""
def dist(point0, point1): return ((point1.x - point0.x)**2 + (point1.y - point0.y)**2)**(1/2)

def diff(fx, t): return (fx(t + DIFF_STEP) - fx(t))/DIFF_STEP
