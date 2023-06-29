from shapes.point import Point
import numpy as np
from settings import *
import sys
import os
from math import sin, cos, asin, sqrt, pi

def dist(point0, point1): return ((point1.x - point0.x)**2 + (point1.y - point0.y)**2)**(1/2)

def diff(fx, t): return (fx(t + DIFF_STEP) - fx(t))/DIFF_STEP

def stopPrint(func, *args, **kwargs):
    with open(os.devnull, "w") as devNull:
        original = sys.stdout
        sys.stdout = devNull
        func(*args, **kwargs)
        sys.stdout = original 

def exp(p, **kwargs):
    rad = kwargs.get("rad", lambda t: 1)
    return Point(rad(p)*np.exp(p.x)*np.cos(p.y), rad(p)*np.exp(p.x)*np.sin(p.y))

def rad_non_centered_ellipse_lambda(a, b, p):
    den = lambda t: a**2*sin(t)**2 + b**2*cos(t)**2
    quad_b = lambda t: a**2*p.y*sin(t) + b**2*p.x*cos(t)
    discr = lambda t: sin(t)**2*(a**2 - p.x**2) + cos(t)**2*(b**2 - p.y**2) + 2*p.x*p.y*sin(t)*cos(t)
    return lambda t: (quad_b(t) + a*b*sqrt(discr(t)))/den(t)

def rad_poly_star_lambda(n, m, k):
    """
    Set k = 1 for straight sides,
    Set m = 1 for regular polygon vertices
    Set n as the number of sides or points
    """
    if abs(k) > 1: raise ValueError("k must be between -1 and 1")
    nom = lambda t: np.cos((2*np.arcsin(k)+pi*m)/(2*n))
    denom = lambda t: np.cos((2*np.arcsin(k*np.cos(n*t.y))+pi*m)/(2*n))
    return lambda t: nom(t)/denom(t)