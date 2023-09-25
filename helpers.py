from shapes import Point
import numpy as np
from settings import *
import sys
import os
from functools import cache
from math import sin, cos, cosh, sinh, sqrt, pi

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

def sin(self): return Point(sin(self.x) * cosh(self.y), cos(self.x) * sinh(self.y))

def cos(self): return Point(cos(self.x) * cosh(self.y), -sin(self.x) * sinh(self.y))

@cache
def betaF(n, m):
    nnn = 2**(n+1) - 1
    if m == 0: return 1.0
    if n > 0 and m < nnn: return 0
    else: return (betaF(n+1, m) - sum([betaF(n, k)*betaF(n, m-k) for k in range(nnn, m-nnn+1)]) - betaF(0, m-nnn))/2

def unit_circle_to_mandelbrot(N):
    def Psi_M(w):
        if w == 0: return 0
        return w + sum(betaF(0, j+1)/(w**j) for j in range(N))
    return lambda w: Psi_M(w)

def flatten_list(input_list):
    result = []
    for item in input_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result