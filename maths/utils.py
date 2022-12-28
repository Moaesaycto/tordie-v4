from shapes.circle import Circle
from utils import color
from svg.svg import Diagram
from shapes.point import Point
from shapes.line import EuclideanLine, PoincareLine
from shapes.parametric import Parametric
import numpy as np

from math import pi
from options import *

def square_grid(N):
    lines = [Parametric(lambda t: t, lambda t: i/N, -SCALE[0]*0.8, SCALE[0]*0.8) for i in range(-N, N+1)]
    lines += [Parametric(lambda t: i/10, lambda t: t, -SCALE[0]*0.8, SCALE[0]*0.8) for i in range(-N, N+1)]
    return lines
